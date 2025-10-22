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

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key="AIzaSyC3FNtIO_binwPV2gN2heIkuVsOJmcKPJg"
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
    
    print(num_of_scene, queries)
    if num_of_scene == 1:
        body = {
            "query": queries[0],
            "top_k": 100,
            "options": 0,
            "model": "pe",
            "isTranslate": True
        }
        res = requests.post("https://nhan6104--search-dev.modal.run", data = json.dumps(body), headers={"Content-Type": "application/json"})

    elif num_of_scene > 1:
        print("taskk 3")
        body = {
            "queries": queries,
            "options": 1,
            "isTranslate": True,
            "top_k": 100,
            "k_path": 3,
            "ban_window": 1,
            "model": "pe"

        }
        res = requests.post("https://nhan6104--searchtask3-dev.modal.run", data = json.dumps(body), headers={"Content-Type": "application/json"})
    
    state["result"] = res.json()
    return Command(update=state, goto="approve_result")



def approve_result(state) -> Command[Literal["accept", "enrich_query"]]:
    approved = interrupt(state["result"])
    print("approve:", approved)  
    return Command(goto="accept" if approved else "enrich_query")

def enrich_query(state) -> Command[Literal["search_query"]]:
    queries = ["Người đầu bếp cho cá vào một tô màu trắng.", "Người đầu bếp đổ bột vào một tô cá để chiên.", "người đầu bếp này dùng đũa để kiểm tra độ nóng của dầu"]
    num_of_scene = len(queries)

    query = Query(number_of_scene=num_of_scene, queries=queries)
    state["enriched_query"].append(query)
    
    return Command(update=state, goto="search_query")


def accept(state):
    return state


graph = StateGraph(state_schema=MessageState)

graph.add_node("search_query", search_query)
graph.add_node("approve_result", approve_result)
graph.add_node("enrich_query", enrich_query)

graph.add_node("accept", accept)

graph.add_edge(START, "search_query")
graph.add_edge("accept", END)


memory = MemorySaver()
app = graph.compile(checkpointer=memory)

message = "Cảnh quay bằng flycam một cây cầu ở TP Hồ Chí Minh, tiếp theo đến cảnh quay tòa nhà Bitexco. Một vài cảnh sau đó chuyển qua quay hình ảnh hồ gươm tại Hà Nội."
query = extract_scene(message)

print(query)

first_result = app.invoke(MessageState(enriched_query = [query]), config={"configurable": {"thread_id": "3"}})

result_accepted = dict()
result_rejected = dict()
while "__interrupt__" in first_result or  "__interrupt__" in result_rejected:
    
    if "__interrupt__" in first_result:
        print(first_result["__interrupt__"])
        first_result = dict()

    elif "__interrupt__" in result_rejected:
        print(result_rejected["__interrupt__"])
        result_rejected = dict()

    message = input("Enter your message: ")
    if message.lower() == "t":
        result_accepted = app.invoke(Command(resume=True), config={"configurable": {"thread_id": "3"}})
        break

    elif message.lower() == "f": 
        result_rejected = app.invoke(Command(resume=False), config={"configurable": {"thread_id": "3"}})
        

print("heloooooooo")

