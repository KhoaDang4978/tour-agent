from tour_intent import msg as graph_app

# Force a hallucinated_price failure — a message likely to make the agent
# state a price without calling the pricing tool, if it's going to fail at all.
# Simpler and more reliable: fake the state directly, bypassing the earlier
# graph nodes, so we test reply_specialist_handoff in isolation first.

from tour_intent import reply_specialist_handoff

fake_state = {
    "message": "How much does the Da Nang tour cost?",
    "hallucinated_price": True,
    "hallucinated_query": False,
}

result = reply_specialist_handoff(fake_state)
print(result)