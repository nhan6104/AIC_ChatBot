from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt, Command
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

def extract_scene(message):
    extract_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract the number of distinct scenes described in the user's input. Extract and list each scene separately. Follow this template: num_of_scene: X \\n scene_1: [description] \\n scene_2: [description] \\n.... \  Only provide the number and the list of scenes without any additional commentary.",
        ),
        ("human", "{query}"),
    ])

    extract_chain = extract_prompt | llm

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


def extract_resource(message):
    enrich_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Base on number of scene from human, please extract exactly number of scene before enrich. Enrich all scenes. Enrich the semantic details emphasizing the objects within each scene to facilitate easier searching by the system. Provide a detailed description for each scene based on the user's input. Following format: Scene:x\nObjects:\[object1:\[list of variety of object1\]\\nobject2:\[list of variety of object2\]\\n....\]\\nContext:\[list of possible contexts\]\\nkeywords:\[list of keysword involve scene\]\n\n... Don't be additional commentary and must follow format, element in list will separate by comma. ",
    ),
    ("human", "{query}"),
    ])

    chain_enrich = enrich_prompt | llm
    ai_msg_enrich = chain_enrich.invoke({
        "query": message
    })

    lst_element_enrich = ai_msg_enrich.content.split("\n\n")
    scenes = []

    for item in lst_element_enrich:
        scene_header, rest = item.split("\n", 1)
        scene_number = int(re.search(r"Scene:(\d+)", scene_header).group(1))
        objects_match = re.search(r"Objects:\[(.*?)\]\s*Context:", rest, re.DOTALL)
        context_match = re.search(r"Context:\[(.*?)\]\s*keywords:", rest, re.DOTALL)
        keywords_match = re.search(r"keywords:\[(.*?)\]", rest, re.DOTALL)

        objects = objects_match.group(1).strip() if objects_match else ""
        context = context_match.group(1).strip() if context_match else ""
        keywords = keywords_match.group(1).strip() if keywords_match else ""
        
        lst_objects = objects.split('\n')
        objects = []
        for obj in lst_objects:
            key, value = obj.split(':', 1)
            objects.append({key.strip(): [v.strip() for v in value.strip().strip('[]').split(',')]})

        scenes.append({
            "scene": int(scene_number),
            "internal_objects": objects,
            "context": context.split(','),
            "keywords": keywords.split(',')
        })

    with open("enrich_scenes.json", "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)



def search_query(state) -> Command[Literal["approve_result"]]:
    enriched_query = state["enriched_query"][-1]
    num_of_scene = enriched_query['number_of_scene']
    queries = enriched_query['queries']
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
            "top_k": 100,
            "k_path": 3,
            "ban_window": 1,
            "model": "pe"

        }
        res = requests.post(f"https://{os.getenv('MODAL_ENPOINT')}--searchtask3-dev.modal.run", data = json.dumps(body), headers={"Content-Type": "application/json"})
    
        result['data'] = res.json()
        result['task_id'] = 3
        state["result"] = result

        return Command(update=state, goto="approve_result")



def approve_result(state) -> Command[Literal["accept", "enrich_query"]]:
    approved = interrupt(state["result"])
    print("approve:", approved)  
    return Command(goto="accept" if approved else "enrich_query")


def enrich_query(state) -> Command[Literal["search_query"]]:
    query = state["enriched_query"][-1]

    old_queries = query['queries']
    num_of_scene = query['number_of_scene']
    
    if num_of_scene > 1:
        query = state["original_query"]
    else:
        query = old_queries[0]
        
    # with open("enrich_scenes.json", "r", encoding="utf-8") as f:
    #     enrich_scenes = json.load(f)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a semantic enrichment model.
            Given the following structured scene data, generate semantically rich, 
            natural language queries that describe the scene from different perspectives or intents.

            - Some queries should sound like a search intent (e.g., "Find drone footage of...")
            - Some should sound like visual descriptions (e.g., "A drone shot showing...")
            - Some should highlight themes or meanings (e.g., "Symbolizing urban development...")
            - All queries must remain faithful to the scene data.
            - Queries should add location or context that scene happened.(e.g., "Security camera footage and police bending down to scan happen in airports", "The barred posts in floodplains, ...")
            - Avoid overly generic queries; be specific to the scene details.

            Respond in English with only sentence. Avoid repeating the same structure. """
        ),
        ("human", "{query}"),
    ])

    enriched_queries = []
    # for query in old_queries:
    enrich_chain = prompt | llm

    ai_msg = enrich_chain.invoke({"query": query})
    enriched_queries.append(ai_msg.content.strip())


    query = Query(number_of_scene=1, queries=enriched_queries)
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
        first_result = self.app.invoke(MessageState(original_query=message, enriched_query = [handled_query]), config={"configurable": {"thread_id": self.thread_id}})
        
        print(first_result)
        data = {
            "thread_id": self.thread_id,
            "data": first_result['result'],
            "message": "Please approve this result"
        }
        return data
    
        # result_accepted = dict()
        # result_rejected = dict()
        # while "__interrupt__" in first_result or  "__interrupt__" in result_rejected:
            
        #     if "__interrupt__" in first_result:
        #         print(first_result["__interrupt__"])
        #         first_result = dict()

        #     elif "__interrupt__" in result_rejected:
        #         print(result_rejected["__interrupt__"])
        #         result_rejected = dict()

        #     message = input("Enter your message: ")
        #     if message.lower() == "t":
        #         result_accepted = self.app.invoke(Command(resume=True), config={"configurable": {"thread_id": self.thread_id}})
        #         break

        #     elif message.lower() == "f": 
        #         result_rejected = self.app.invoke(Command(resume=False), config={"configurable": {"thread_id": self.thread_id}})
    
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
            "data": result_rejected['result'],
            "message": "Please approve this result"

        }
        return data
    


message = """Một cảnh quay từ camera an ninh về một lần khám xét.
E1: Người cảnh sát vẫy tay đưa nghi phạm vào phòng.
E2: Sau khi vào phòng, cảnh sát ra hiệu chỉ tay vào vị trí cần đứng. Hãy lấy cảnh đầu tiên chỉ tay được thực hiện
E2: Khoảnh khắc đầu tiên cảnh sát hoàn toàn khom người xuống để quét kiểm tra"""



        

print("heloooooooo")

