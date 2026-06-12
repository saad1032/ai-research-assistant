"""
Document Ingestion module for processing medical reports (PDFs, etc.).
Responsible for loading documents, splitting them into text chunks, and embedding them into the vector store.
"""

import os
from typing import List, Tuple
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts raw text from a PDF file using the pypdf library.

    Args:
        file_path (str): The absolute or relative path to the PDF file.

    Returns:
        str: The extracted raw text from all pages of the PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    return text

def load_and_split_document(file_path: str) -> Tuple[List[Document], List[Document]]:
    """
    Loads a medical report PDF, extracts its text, and splits it using two strategies:
    1. Semantic Strategy: RecursiveCharacterTextSplitter (chunk_size=512, chunk_overlap=64).
    2. Fixed Strategy: Fixed-size chunking (chunk_size=512, chunk_overlap=0).

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        Tuple[List[Document], List[Document]]: A tuple containing:
            - List of semantic chunks as LangChain Document objects.
            - List of fixed-size chunks as LangChain Document objects.
    """
    filename = os.path.basename(file_path)
    
    # 1. Extract text from the PDF
    raw_text = extract_text_from_pdf(file_path)
    
    if not raw_text.strip():
        raise ValueError(
            "The uploaded PDF contains no selectable text. It appears to be scanned or image-based. "
            "Please upload a text-based PDF report."
        )
    
    # 2. Semantic Chunking Strategy
    semantic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64
    )
    semantic_texts = semantic_splitter.split_text(raw_text)
    
    semantic_chunks = []
    for idx, chunk_text in enumerate(semantic_texts):
        doc = Document(
            page_content=chunk_text,
            metadata={
                "source": filename,
                "chunk_index": idx,
                "strategy": "semantic"
            }
        )
        semantic_chunks.append(doc)
        
    # 3. Fixed-size Chunking Strategy (chunk_size=512, zero overlap)
    fixed_chunks = []
    chunk_index = 0
    for i in range(0, len(raw_text), 512):
        chunk_text = raw_text[i:i + 512]
        # Ignore empty chunks
        if not chunk_text.strip():
            continue
        doc = Document(
            page_content=chunk_text,
            metadata={
                "source": filename,
                "chunk_index": chunk_index,
                "strategy": "fixed"
            }
        )
        fixed_chunks.append(doc)
        chunk_index += 1

    return semantic_chunks, fixed_chunks

def ingest_document(file_path: str) -> None:
    """
    Extracts, splits, and embeds document contents in both semantic and fixed-size ChromaDB collections.

    Args:
        file_path (str): The path to the PDF file to ingest.
    """
    semantic_chunks, fixed_chunks = load_and_split_document(file_path)
    
    from app.retriever import build_vectorstore
    build_vectorstore(semantic_chunks, "reports_semantic")
    build_vectorstore(fixed_chunks, "reports_fixed")


