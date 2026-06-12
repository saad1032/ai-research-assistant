"""
LLM and RAG Orchestration module.
Sets up the QA/RAG response generation using OpenAI API and computes a hallucination guard score.
"""

import os
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client (will read OPENAI_API_KEY from env)
# If key is missing, it will raise an error when a call is made, which we will handle.
openai_client = OpenAI()

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Fetches embeddings from OpenAI for a list of texts using the text-embedding-3-small model.
    """
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two numpy vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return float(dot_product / (norm_vec1 * norm_vec2))

def generate_response(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates an answer to the query using gpt-3.5-turbo based ONLY on the retrieved chunks.
    Performs a hallucination guard check using cosine similarity on OpenAI embeddings.

    Args:
        query (str): The user's question.
        retrieved_chunks (List[Dict[str, Any]]): The chunks retrieved from the vector store.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "answer": The string answer from the LLM.
            - "confidence_score": Average similarity between the answer and chunks.
            - "is_flagged": True if confidence_score is below 0.72 ("low confidence").
            - "source_chunks": List of the retrieved chunks used.
    """
    # 1. Construct context from retrieved chunks
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(f"Content: {chunk['content']}\nSource: {chunk['metadata'].get('source', 'Unknown')}")
    context = "\n\n---\n\n".join(context_parts)

    # 2. Build system and user prompts
    system_prompt = (
        "You are an expert medical report analyzer. Your task is to answer the user's questions "
        "based ONLY on the provided medical report context. "
        "Strictly adhere to the following rules:\n"
        "1. Answer the question using ONLY facts directly mentioned in the context.\n"
        "2. If the answer cannot be found in the provided context, you MUST state exactly: 'I cannot find this in the report'.\n"
        "3. Do not make up any medical statements, and do not extrapolate or use external medical knowledge."
    )

    user_prompt = (
        f"Medical Report Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "Answer:"
    )

    # 3. Call OpenAI API
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error generating answer from OpenAI: {str(e)}"
        return {
            "answer": answer,
            "confidence_score": 0.0,
            "is_flagged": True,
            "source_chunks": retrieved_chunks
        }

    # 4. Compute Hallucination Guard Score
    # Special case: If LLM states it cannot find the information, we don't flag it as hallucination
    if "I cannot find this in the report" in answer or not retrieved_chunks:
        return {
            "answer": answer,
            "confidence_score": 1.0,
            "is_flagged": False,
            "source_chunks": retrieved_chunks
        }

    # Encode answer and chunks
    try:
        chunk_contents = [chunk["content"] for chunk in retrieved_chunks]
        # Batch embed the answer and all chunks
        all_texts = [answer] + chunk_contents
        all_embeddings = get_embeddings(all_texts)
        
        answer_embedding = np.array(all_embeddings[0])
        chunk_embeddings = [np.array(emb) for emb in all_embeddings[1:]]

        similarities = []
        for chunk_emb in chunk_embeddings:
            sim = compute_cosine_similarity(answer_embedding, chunk_emb)
            similarities.append(sim)

        avg_similarity = float(np.mean(similarities)) if similarities else 0.0
    except Exception:
        # Fallback if encoding fails
        avg_similarity = 0.0

    # Determine if flagged (threshold < 0.72)
    is_flagged = avg_similarity < 0.72

    return {
        "answer": answer,
        "confidence_score": round(avg_similarity, 4),
        "is_flagged": is_flagged,
        "source_chunks": retrieved_chunks
    }


