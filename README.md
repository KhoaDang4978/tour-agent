# Tour Agent

A booking agent for an imaginary tour operator, built to practice LangGraph, RAG,
and evaluation patterns. This is a learning project, not production code — it
does not connect to a real business.

## Capabilities

- **Destination classification** — extracts intent from a customer message and
  routes it into one of several scopes (in scope, out of scope, ambiguous,
  unsupported route, etc.)
- **Pricing lookup** — answers questions about tour cost using a fixed pricing
  table
- **Policy Q&A via RAG** — answers questions about cancellation, pickup,
  payment, and other policies by retrieving relevant text from a small vector
  database, rather than relying on the model's own guesses

## RAG Architecture

The policy system runs in two phases.

**Setup phase** (`policy_embed.py`) — run once to build the vector database:

```
Policy text (POLICY dict)
    → split into individual sentence chunks
    → each chunk embedded (NVIDIA embeddings)
    → stored in ChromaDB
```

**Query phase** (runs on every customer question):

```
Customer question
    → embedded (same embedding model as the chunks)
    → search ChromaDB for the closest matching chunks
    → best match retrieved
    → injected into the agent's prompt
    → agent responds using the retrieved text
```

If the closest match is too dissimilar to the question (distance above a set
threshold), the tool returns "no relevant policy found" instead of returning a
weak match as if it were a good one.

## Setup

1. Clone or download this repository.
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set the required environment variables (in your shell profile, e.g.
   `~/.zshrc`):
   ```bash
   export DEEPSEEK_API_KEY="your_key_here"
   export NVIDIA_API_KEY="your_key_here"
   export LANGSMITH_TRACING=true
   export LANGSMITH_API_KEY="your_key_here"
   export LANGSMITH_PROJECT="tourbot"
   ```
5. Build the vector database (run once, before anything else will work):
   ```bash
   python3 policy_embed.py
   ```
6. Run the agent:
   ```bash
   python3 tour_bot.py
   ```
