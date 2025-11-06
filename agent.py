from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt, Command
from promptTemplate import ENRICH_PROMPT_TEMPLATE, EXTRACT_PROMPT_TEMPLATE, FEWSHOT_PROMPT_TEMPLATE
import requests
import json
import re
from dotenv import load_dotenv
load_dotenv()
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv('GOOGLE_API_KEY')
)

class Query(TypedDict):
    number_of_scene: int
    queries:  List[str]

class MessageState(TypedDict):
    original_query: str
    enriched_query: List[Query]
    result: any
    top_k_task3: List[int]
    top_k_task3_index: int
    search_input: Query
    mode: int

def extract_scene(message):

    extract_chain = EXTRACT_PROMPT_TEMPLATE | llm

    ai_msg = extract_chain.invoke({
        "query": message
    })
    # lst_res = ['num_of_scene: 3', 'scene_1: Cảnh quay bằng flycam một cây cầu ở TP Hồ Chí Minh,', 'scene_2: cảnh quay tòa nhà Bitexco,', 'scene_3: quay hình ảnh hồ gươm tại Hà Nội']
    query = Query(queries=[])
    
    if ai_msg.content:
        lst_res = ai_msg.content.split("\n")
        res_dict = {}
        for item in lst_res:
            key, value = item.split(":", 1)
            if key == 'num_of_scene':
                query['number_of_scene'] = int(value.strip())
                res_dict[key.strip()] = int(value.strip())
            else:
                query['queries'].append(value.strip())
                res_dict[key.strip()] = value.strip()

    elif not ai_msg.content:
        query['number_of_scene'] = 1
        query['queries'].append(message)
    
    return query


def search_query(state) -> Command[Literal["approve_result", "search_query"]]:
    query = state["search_input"]
    num_of_scene = query['number_of_scene']
    queries = query['queries']
    result = dict()

    print(num_of_scene, queries)
    if num_of_scene == 1:
        body = {
            "query": queries[0],
            "top_k": 10,
            "options": 0,
            "model": "pe",
            "isTranslate": True
        }
        res = requests.post(f"https://{os.getenv('MODAL_ENPOINT')}--search-dev.modal.run", data = json.dumps(body), headers={"Content-Type": "application/json"})
        
        result['data'] = res.json()
        result['task_id'] = 1
        state["result"] = result

        return Command(update=state, goto="approve_result")

    elif num_of_scene > 1:        
        body = {
            "queries": queries,
            "options": 0,
            "isTranslate": True,
            "top_k": state["top_k_task3"][state["top_k_task3_index"]],
            "k_path": 3,
            "ban_window": 1,
            "model": "pe"

        }
        res = requests.post(f"https://{os.getenv('MODAL_ENPOINT')}--searchtask3-dev.modal.run", data = json.dumps(body), headers={"Content-Type": "application/json"})
    
        result['data'] = res.json()
        result['task_id'] = 3
        state["result"] = result

        if len(result['data']) == 0:
            if state["top_k_task3_index"] < len(state["top_k_task3"]):
                state["top_k_task3_index"] += 1

            if state["top_k_task3_index"] == len(state["top_k_task3"]): 
                state["top_k_task3_index"] = len(state["top_k_task3"]) - 1

            return Command(update=state, goto="search_query")


        return Command(update=state, goto="approve_result")


   
def approve_result(state) -> Command[Literal["accept", "enrich_query"]]:
    approved = interrupt(state["result"])
    print("approve:", approved)  
    return Command(goto="accept" if approved else "enrich_query")


def enrich_query(state) -> Command[Literal["search_query", "enrich_query"]]:

    if state["mode"] == 1:
        if state["top_k_task3_index"] < len(state["top_k_task3"]) - 1:
            state["top_k_task3_index"] += 1
            return Command(update=state, goto="search_query")
        
        if state["top_k_task3_index"] == len(state["top_k_task3"]) - 1:
            state["top_k_task3_index"] = -1  
            state["mode"] = 2    
            return Command(update=state, goto="enrich_query")
        
    if state["mode"] == 2:
        original_query = state["original_query"]
        few_show_chain = FEWSHOT_PROMPT_TEMPLATE | llm
        ai_msg = few_show_chain.invoke({"query": original_query})

        query = Query(number_of_scene=1, queries=[ai_msg.content.strip()])
        state["enriched_query"].append(query)

        state["search_input"] = query
        state["mode"] = 3

        return Command(update=state, goto="search_query")
    
    if state["mode"] == 3:
        query = state["enriched_query"][0] # Lấy query gốc đầu tiên để làm giàu
        old_queries = query['queries']
        num_of_scene = query['number_of_scene']
        
        print(old_queries, num_of_scene)

        if num_of_scene > 1:
            if state["top_k_task3_index"] == -1:
                query = " ".join(old_queries)

                enrich_chain = ENRICH_PROMPT_TEMPLATE | llm 
                ai_msg = enrich_chain.invoke({"query": query})
                query = extract_scene(ai_msg.content.strip())
                
                state["search_input"] = query
                state["enriched_query"].append(query)

            if state["top_k_task3_index"] < len(state["top_k_task3"]):
                state["top_k_task3_index"] += 1


            if state["top_k_task3_index"] == len(state["top_k_task3"]):
                state["top_k_task3_index"] = -1
                return Command(update=state, goto="enrich_query")


            return Command(update=state, goto="search_query")
        
        else:

            query = old_queries[0]
            
            enrich_chain = ENRICH_PROMPT_TEMPLATE | llm 
            ai_msg = enrich_chain.invoke({"query": query})
            # enriched_queries.append(ai_msg.content.strip())
            query = extract_scene(ai_msg.content.strip())
            
            state["search_input"] = query
            state["enriched_query"].append(query)
        
            return Command(update=state, goto="search_query")


def accept(state):
    return state


class Agent:
    def __init__(self):
        memory = MemorySaver()
        graph = self.graph_define()
        self.app = graph.compile(checkpointer=memory)
        self.thread_id = 0

    def graph_define(self):
        graph = StateGraph(state_schema=MessageState)

        graph.add_node("search_query", search_query)
        graph.add_node("approve_result", approve_result)
        graph.add_node("enrich_query", enrich_query)

        graph.add_node("accept", accept)

        graph.add_edge(START, "search_query")
        graph.add_edge("accept", END)
        
        return graph

    def searchQuery(self, query):
        handled_query = extract_scene(query)

        top_k_task3 = [100, 300, 600, 900, 1024, 2048]

        message_state = MessageState(
            original_query = query, 
            search_input = handled_query,
            enriched_query = [handled_query],
            top_k_task3 = top_k_task3,
            top_k_task3_index = 4,
            mode = 1
        )
        
        first_result = self.app.invoke(message_state, config={"configurable": {"thread_id": self.thread_id}})
        
        data = {
            "thread_id": self.thread_id,
            "data": first_result['result'],
            "mode": first_result["mode"],
            "top_k_task3_index": first_result["top_k_task3_index"],
            "message": "Please approve this result"
        }
        return data
    
      
    def approvalTrue(self):
        result_accepted = self.app.invoke(Command(resume=True), config={"configurable": {"thread_id": self.thread_id}})
        self.thread_id += 1
        data = {
            "thread_id": self.thread_id,
            "data": result_accepted,
            "message": "New session will be set"
        }
        return data

    def approvalFalse(self):
        result_rejected = self.app.invoke(Command(resume=False), config={"configurable": {"thread_id": self.thread_id}})
        data = {
            "thread_id": self.thread_id,
            "mode": result_rejected["mode"],
            "top_k_task3_index": result_rejected["top_k_task3_index"],
            "data": result_rejected['result'],
            "message": "Please approve this result"

        }
        return data
    


# message = """Một cảnh quay từ camera an ninh về một lần khám xét.
# E1: Người cảnh sát vẫy tay đưa nghi phạm vào phòng.
# E2: Sau khi vào phòng, cảnh sát ra hiệu chỉ tay vào vị trí cần đứng. Hãy lấy cảnh đầu tiên chỉ tay được thực hiện
# E2: Khoảnh khắc đầu tiên cảnh sát hoàn toàn khom người xuống để quét kiểm tra"""



        

print("heloooooooo")

