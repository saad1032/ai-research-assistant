"""
FastAPI application entrypoint for the Medical Report Analyzer.
Responsible for setting up endpoints to upload files, trigger ingestion, and query the RAG model.
"""

import os
import shutil
from typing import Literal, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field

# Local module imports
from app.ingestor import load_and_split_document
from app.retriever import build_vectorstore, retrieve, compare_strategies
from app.llm import generate_response

app = FastAPI(
    title="Medical Report Analyzer API",
    description="Backend API for RAG-based analysis of medical lab reports",
    version="0.1.0"
)

# Pydantic models for structured requests and responses
class AskRequest(BaseModel):
    query: str = Field(..., description="The medical question or search query.")
    strategy: Literal["semantic", "fixed"] = Field(
        "semantic", 
        description="The chunking strategy to use for retrieval: 'semantic' or 'fixed'."
    )

class ChunkMetadata(BaseModel):
    source: str
    chunk_index: int
    strategy: str

class ChunkResponse(BaseModel):
    id: str
    content: str
    metadata: ChunkMetadata
    similarity_score: float

class AskResponse(BaseModel):
    answer: str
    confidence_score: float
    is_flagged: bool
    source_chunks: List[ChunkResponse]

class CompareResponse(BaseModel):
    semantic: List[ChunkResponse]
    fixed: List[ChunkResponse]

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Medical Report Analyzer API",
        "endpoints": {
            "POST /upload": "Upload a medical report PDF for ingestion",
            "POST /ask": "Ask a question about the reports using a specific chunking strategy",
            "GET /compare": "Compare search results from both strategies side by side",
            "GET /health": "Get API health status"
        }
    }

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Returns the health status of the application."""
    return {"status": "ok"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accepts a PDF file upload, splits it using semantic and fixed strategies,
    and builds corresponding vector stores.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Create the data folder inside the project root if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, file.filename)

    # Save uploaded file to the data directory
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    # Trigger ingestion and vector store construction
    try:
        semantic_chunks, fixed_chunks = load_and_split_document(file_path)
        
        # Build both ChromaDB collections
        build_vectorstore(semantic_chunks, "reports_semantic")
        build_vectorstore(fixed_chunks, "reports_fixed")
        
        return {
            "message": f"Successfully processed and ingested '{file.filename}'.",
            "filename": file.filename,
            "semantic_chunks_count": len(semantic_chunks),
            "fixed_chunks_count": len(fixed_chunks)
        }
    except Exception as e:
        # Clean up file in case of failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to ingest and split PDF: {str(e)}"
        )

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> Dict[str, Any]:
    """
    Accepts a query and a strategy, retrieves relevant chunks from the corresponding
    collection, and generates an answer using the LLM with hallucination guarding.
    """
    collection_name = "reports_semantic" if request.strategy == "semantic" else "reports_fixed"

    # 1. Retrieve top-k chunks
    try:
        retrieved_chunks = retrieve(request.query, collection_name, top_k=5)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")

    # 2. Query LLM with retrieved context
    try:
        response = generate_response(request.query, retrieved_chunks)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

@app.get("/compare", response_model=CompareResponse)
async def compare_query(query: str = Query(..., description="Query to test against both strategies.")) -> Dict[str, Any]:
    """
    Accepts a query and returns retrieved chunks from both 'semantic' and 'fixed'
    strategies for comparison.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query parameter cannot be empty.")

    try:
        return compare_strategies(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

