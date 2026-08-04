import chromadb
from chromadb.utils.embedding_functions import ChromaLangchainEmbeddingFunction
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from policy_chunk import document_metadata, document_ids, document_texts

nvidia_embeddings = NVIDIAEmbeddings(model="nvidia/llama-nemotron-embed-1b-v2")
chroma_ef = ChromaLangchainEmbeddingFunction(nvidia_embeddings)

client = chromadb.PersistentClient(path="./chroma_policies")
collection = client.get_or_create_collection(
    name="tour_policies",
    embedding_function=chroma_ef,
)
collection.add(
    documents=document_texts,
    ids=document_ids,
    metadatas=document_metadata,
)

query_vector = nvidia_embeddings.embed_query("does the tour allow pets")

results = collection.query(
    query_embeddings=[query_vector],
    n_results=2,
    include=["documents", "distances"]
)

print(results["distances"])