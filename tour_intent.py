from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import TypedDict
from tour_bot import get_agent_response
from tour_bot import extract_tour_intent
import json
import re
from tour_state import fetch_id_exists, get_customer_context
import chromadb
from chromadb.utils.embedding_functions import ChromaLangchainEmbeddingFunction
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from tour_bot import llm, get_tour_policy, get_tour_pricing
from langchain.agents import create_agent

nvidia_embeddings = NVIDIAEmbeddings(model="nvidia/llama-nemotron-embed-1b-v2")
chroma_ef = ChromaLangchainEmbeddingFunction(nvidia_embeddings)


DECLINE_MSG = "Xin lỗi, Tour Travel hiện tại chỉ phục vụ các tour: Đà Nẵng, Nha Trang, Sài Gòn Chúng tôi chưa có tour tại {destination}."
AMBIGUOUS_MSG = "Bạn muốn tham gia tour nào? {candidates}"

class GraphState(TypedDict):
    message: str
    scope: str
    reply: str
    tour: str
    candidates: str
    tools_called: list
    chat_id: str
    eval_result: str
    has_policy_intent: bool
    hallucinated_price: bool
    hallucinated_query: bool


base_system_prompt = """
    You are a specialist in handling edge cases. Your job is to handle failed evaluated replies based on which kind it is.

    You must NOT state a price without calling 
    get_tour_pricing, must NOT state a policy without calling 
    get_tour_policy.

    If you still cannot get an answer, say so honestly, do NOT guess.
    """

hallucinated_price_prompt = """

    This is a price hallucination failure. Get the correct price from the tool again.
    """

hallucinated_query_prompt = """

    This is a query hallucination failure. Get the correct policy again.
    """

def reply_specialist_handoff(state: GraphState) -> dict:
    org_customer_msg = state["message"]
    hallucinated_price = state["hallucinated_price"]
    hallucinated_query = state["hallucinated_query"]

    full_prompt = base_system_prompt
    if hallucinated_price:
        full_prompt += hallucinated_price_prompt
    if hallucinated_query:
        full_prompt += hallucinated_query_prompt
    history = [("user", org_customer_msg)]
    agent = create_agent(
        model=llm,
        tools=[get_tour_pricing, get_tour_policy],
        system_prompt=full_prompt
    )

    response = agent.invoke({"messages": history})
    agent_reply = response["messages"][-1].content
    return {"reply":agent_reply}




def extract_intent(state: GraphState) -> dict:
    context = ""
    if fetch_id_exists(state["chat_id"]):
        context = f"Here's the historical profile of the customer: {get_customer_context(state['chat_id'])}"
    
    intent = extract_tour_intent.invoke({"message": state["message"], "context": context})
    parsed = json.loads(intent)
    return {
        "scope": parsed["scope"],
        "tour": parsed.get("tour"),
        "candidates": parsed.get("candidates"),
        "has_policy_intent": parsed.get("has_policy_intent"),
    }

def reply_handoff_to_human(state: GraphState) -> dict:
    reply = "Xin lỗi, tôi chưa hiểu rõ yêu cầu của bạn. Một nhân viên Tour Travel sẽ liên hệ với bạn sớm nhất."
    return {"reply": reply}

def evaluate_reply(state: GraphState) -> dict:
    reply = state["reply"]
    tools_called = state["tools_called"]
    has_policy_intent = state["has_policy_intent"]
    high_distance = False 

    if has_policy_intent:
        client = chromadb.PersistentClient(path="./chroma_policies")
        collection = client.get_or_create_collection(name="tour_policies")
        query_vector = nvidia_embeddings.embed_query(reply)
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=2,
            include=["documents", "distances"]
        )
        for x in result["distances"][0]:
            if x > 1.6:
                high_distance = True

    has_price_pattern = bool(re.search(r"\d{1,3}(,\d{3})+", reply))
    hallucinated_price = has_price_pattern and not tools_called
    hallucinated_query = high_distance and not tools_called

    if hallucinated_price or hallucinated_query:
        return {
            "eval_result": "fail",
            "hallucinated_price": hallucinated_price,
            "hallucinated_query": hallucinated_query
        }
    return {
        "eval_result": "pass",
        "hallucinated_price": False,
        "hallucinated_query": False
        }

def route_by_eval(state: GraphState) -> str:
    return state["eval_result"]

def reply_in_scope(state: GraphState) -> dict:
    reply, tools_called = get_agent_response(state["message"])
    return {"reply": reply, "tools_called": tools_called}

def reply_out_of_scope(state: GraphState) -> dict:
    reply = DECLINE_MSG(tour=state.get("destination", "khu vực này"))
    return {"reply": reply}

def reply_no_destination(state: GraphState) -> dict:
    reply, tools_called = get_agent_response(state["message"])
    return {"reply": reply, "tools_called": tools_called}

def reply_ambiguous(state: GraphState) -> dict:
    reply = AMBIGUOUS_MSG(candidates=", ".join(state.get("candidates", [])))
    return {"reply": reply}

def route_by_scope(state: GraphState) -> str:
    return state["scope"]

graph = StateGraph(GraphState)
graph.add_node("extract_intent", extract_intent)
graph.add_node("reply_specialist_handoff", reply_specialist_handoff)
graph.add_node("reply_in_scope", reply_in_scope)
graph.add_node("reply_out_of_scope", reply_out_of_scope)
graph.add_node("reply_no_destination", reply_no_destination)
graph.add_node("reply_ambiguous", reply_ambiguous)
graph.add_node("evaluate_reply", evaluate_reply)

graph.add_edge(START, "extract_intent")
graph.add_conditional_edges(
    "extract_intent",
    route_by_scope,
    {   "in_scope": "reply_in_scope",
        "out_of_scope": "reply_out_of_scope",
        "no_destination": "reply_no_destination",
        "ambiguous": "reply_ambiguous"
    }
)
graph.add_edge("reply_in_scope", "evaluate_reply")
graph.add_conditional_edges(
    "evaluate_reply",
    route_by_eval,
    {   "pass": END,
        "fail": "reply_specialist_handoff"
    }
)
graph.add_edge("reply_out_of_scope", END)
graph.add_edge("reply_no_destination", END)
graph.add_edge("reply_ambiguous", END)
graph.add_edge("reply_specialist_handoff", END)

msg = graph.compile()


