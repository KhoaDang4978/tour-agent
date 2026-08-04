from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import chromadb
import os
import json
from enum import Enum
from typing import List
from pydantic import BaseModel
from flask import request
from chromadb.utils.embedding_functions import ChromaLangchainEmbeddingFunction
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

nvidia_embeddings = NVIDIAEmbeddings(model="nvidia/llama-nemotron-embed-1b-v2")
chroma_ef = ChromaLangchainEmbeddingFunction(nvidia_embeddings)

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "tour_pricing.json")

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    extra_body={"thinking": {"type": "disabled"}}
)

class TourPriceResult(BaseModel):
    tour: str
    price_vnd: int
    found: bool

class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ScopeResult(str, Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"
    NO_DESTINATION = "no_destination"

class TourIntent(BaseModel):
    tour: str | None
    scope: ScopeResult
    candidates: list[str] | None
    confidence: Confidence
    has_policy_intent: bool

@tool
def get_tour_policy(question: str) -> str:
    """Get the tour policies.

    Args:
        question: customer's question about tour policies

    Returns:
        Policy as a string, or an error message explaining what went wrong.
    
    Use this tool for questions about policy/logistic/cancellation.
    Do NOT use this tour for questions about pricing.
    """

    try:
        client = chromadb.PersistentClient(path="./chroma_policies")
        collection = client.get_or_create_collection(name="tour_policies", embedding_function=chroma_ef)

        query_vector = nvidia_embeddings.embed_query(question)

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=2,
            include=["documents", "distances"]
        )

        if result["distances"][0][0] > 1.6:
            return("No relevant policy found")
        
        sentences = result["documents"][0]
        return " ".join(sentences)

    except FileNotFoundError:
        return "Policy data unavailable"
    except Exception as e:
        return f"Error retrieving policy: {str(e)}"

    
@tool
def extract_tour_intent(message: str, context = "") -> str:
    """Extract the customer's intended destination and determine if it's in Tour Travel's service area.
    
    Args:
        message: The customer's raw message text.
        context: Optional prior customer history, already looked up by the caller.
    
    Returns:
        JSON string of TourIntent (tour, scope, candidates, confidence, has_policy_intent).
    """
    try:
        system_prompt = """Your job is to extract the customer's intended destination and determine 
                        if it's in Tour Travel's service area.
                        The 4 scopes:
                        - In scope: a single served destination.
                        - Out of scope: a single destination that is NOT served (e.g. "tour  
                        to Vung Tau " -> out_of_scope, destination="Da Nang"). This applies even if 
                         only one unserved city is mentioned.
                        - Ambiguous: destination unclear, multiple served candidates possible 
                        (e.g. "tour to the landmark").
                        - No destination: No destination mentioned at all or the destination text doesn't correspond to any real place.

                        Critical distinction: a single unserved tour is ALWAYS out_of_scope, never 
                        unsupported_route. unsupported_route requires multiple named destinations.

                        Our supported tours: Da Nang, Nha Trang, Saigon.
                        
                        For the has_policy_intent field:

                        This field is for identifying whether the customer has any question about the tour policies.

                        One example where the message is policy related: "Does the tour allow pets?"

                        One example where the message is NOT policy related: "How much does a Da Nang tour cost?"

                        """
    
        full_prompt = system_prompt + context
        structured_llm = llm.with_structured_output(TourIntent, method="function_calling")
        result = structured_llm.invoke(f"{full_prompt}\n\nCustomer message: {message}")
        return result.model_dump_json()
    
    except Exception as e:
        return f"Error extracting intention: {str(e)}"
@tool
def get_tour_pricing(tour: str) -> str:
    """Get the tour price for a tour.
    Args:
        tour: Destination only. Valid options: 'da nang', 'nha trang', 'saigon'
                                Example: 'da nang'
    
    Returns:
        Price as string, or error message explaining what went wrong.
    
    Use this tool when customer asks about pricing.
    Do NOT use it for tour details or time related questions.
    """
    try:
        with open("tour_pricing.json") as f:
            data = json.load(f)
        key = f"{tour.lower()}"
        price = data.get(key)

        if not price:
            result = TourPriceResult(tour=tour, price_vnd=0, found=False)
            return result.model_dump_json()
        price_number = int(price.replace(",", "").replace(" VND", ""))
        result = TourPriceResult(tour=tour, price_vnd=price_number, found=True)
        return result.model_dump_json()
    
    except FileNotFoundError:
        return "Pricing data unavailable. Please contact support."
    except Exception as e:
        return f"Error retrieving price: {str(e)}"
    
base_system_prompt = """
    You are the booking assistant for Tour Travel, a tour operationa and travel ecosystem in Vietnam. Your primary objective is to assist customer with tour bookings"

    Available tour destination: Da Nang,  Nha Trang and Saigon.
"""

def get_agent_response(user_input: str) -> tuple[str, list[str]]:
    history = [("user", user_input)]
    agent = create_agent(
        model=llm,
        tools=[get_tour_pricing, get_tour_policy],
        system_prompt=base_system_prompt,
        checkpointer=InMemorySaver()
    )
    thread_config = {"configurable": {"thread_id": "1"}}
    response = agent.invoke({"messages": history}, thread_config)
    agent_reply = response["messages"][-1].content
    
    tool_names = []
    for message in response["messages"]:
        calls = getattr(message, "tool_calls", [])
        for call in calls:
            tool_names.append(call["name"])
    
    return agent_reply, tool_names
