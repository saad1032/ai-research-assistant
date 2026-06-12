"""
Retriever module for querying ChromaDB.
Sets up the vector store retriever with OpenAI embeddings to bypass local DLL load policies.
"""

import os
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Define vector store directory relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# Initialize the OpenAI Embedding function
# Using text-embedding-3-small as a modern, cost-effective embedding model
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key if openai_api_key else "dummy_key",
    model_name="text-embedding-3-small"
)

def build_vectorstore(chunks: List[Document], collection_name: str) -> None:
    """
    Embeds and stores LangChain Document chunks in a persistent ChromaDB collection.

    Args:
        chunks (List[Document]): A list of LangChain Document objects.
        collection_name (str): The name of the collection to store the chunks in.
    """
    if not chunks:
        return

    # Initialize the Persistent Client
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    
    # Get or create the collection (using cosine similarity space)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        # Ensure metadata contains only allowable types (str, int, float, bool)
        metadatas.append(chunk.metadata)
        
        # Build unique, deterministic IDs
        source = chunk.metadata.get("source", "unknown")
        strategy = chunk.metadata.get("strategy", "unknown")
        chunk_idx = chunk.metadata.get("chunk_index", idx)
        ids.append(f"{source}_{strategy}_{chunk_idx}")
        
    # Add documents to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

def retrieve(query: str, collection_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Queries a collection and returns the top-k most similar chunks with cosine similarity.

    Args:
        query (str): The search query.
        collection_name (str): Name of the ChromaDB collection to search.
        top_k (int): Number of top results to return.

    Returns:
        List[Dict[str, Any]]: List of dictionary results containing 'id', 'content', 'metadata', and 'similarity_score'.
    """
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name not in existing_collections:
        raise ValueError(
            f"Collection '{collection_name}' does not exist. Please run ingestion for this strategy first."
        )
        
    collection = client.get_collection(
        name=collection_name,
        embedding_function=openai_ef
    )

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    formatted_results = []
    if results and "documents" in results and results["documents"]:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [1.0] * len(documents)
        ids = results["ids"][0]
        
        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # In HNSW cosine space, distance is (1 - cosine_similarity).
            # Convert to similarity score:
            similarity = 1.0 - dist
            formatted_results.append({
                "id": doc_id,
                "content": doc,
                "metadata": meta,
                "similarity_score": similarity
            })
            
    return formatted_results

def compare_strategies(query: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves and compares chunks for the given query from both semantic and fixed strategies.

    Args:
        query (str): The search query.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary comparing results:
            {
                "semantic": List of retrieved semantic chunks,
                "fixed": List of retrieved fixed chunks
            }
    """
    comparison = {
        "semantic": [],
        "fixed": []
    }
    
    # Query semantic collection
    try:
        comparison["semantic"] = retrieve(query, "reports_semantic")
    except ValueError:
        # Handle case where collection doesn't exist yet
        comparison["semantic"] = []
        
    # Query fixed collection
    try:
        comparison["fixed"] = retrieve(query, "reports_fixed")
    except ValueError:
        # Handle case where collection doesn't exist yet
        comparison["fixed"] = []
        
    return comparison


