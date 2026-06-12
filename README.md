# Medical Report Analyzer 🩺

A RAG-based clinical document analyzer comparing semantic and fixed-size chunking strategies with hallucination guarding.

[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](#)
[![HuggingFace Space](https://img.shields.io/badge/Space-HuggingFace-orange)](#)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue)](#)

---

## 🔍 Why This Project?

Medical lab reports are rich in dense, highly critical health metrics that patients and practitioners often struggle to interpret quickly. General-purpose LLMs struggle with precision and are prone to hallucinations when retrieving context from raw, unformatted clinical reports. This project addresses the challenge by implementing a localized Retrieval-Augmented Generation (RAG) system that compares chunking methodologies side by side and applies similarity thresholds to detect and flag fabricated model responses.

---

## 📊 Benchmark Results

| Evaluation Metric | Semantic Strategy (Recursive) | Fixed-Size Strategy (Slicing) |
| :--- | :---: | :---: |
| **Answer Relevance (RAGAS)** | - | - |
| **Context Precision** | - | - |
| **Avg Retrieval Latency** | - | - |
| **Hallucination Rate** | - | - |

---

## 🏗️ System Architecture

```mermaid
graph LR
    A[PDF] --> B[PyPDF Extractor]
    B --> C[Chunker (Semantic/Fixed)]
    C --> D[OpenAI Embedder]
    D --> E[ChromaDB]
    E --> F[Retriever]
    F --> G[LLM (GPT-3.5)]
    G --> H[Streamlit UI]
```

---

## 🛠️ Design Decisions

*   **Why Semantic Chunking Outperforms Fixed**: Semantic chunking (`RecursiveCharacterTextSplitter`) groups text by prioritizing logical structural boundaries like double newlines, single newlines, and spaces. This prevents splitting clinical sentences or breaking medical metrics (e.g., separating `Hemoglobin: 14` from its units or normal ranges). Fixed-size chunking slices text at arbitrary offsets, often dividing critical values across boundaries, causing retrieval failures.
*   **How the Hallucination Guard Works**: The hallucination guard computes a localized semantic verification score. Once `gpt-3.5-turbo` yields an answer, the response text is re-embedded using OpenAI's `text-embedding-3-small` API. The system calculates the average cosine similarity between this response vector and each retrieved context document. If the average falls below `0.72`, the Streamlit UI flags the result as **Low Confidence**.
*   **Why ChromaDB was Chosen over Pinecone**: ChromaDB runs as a lightweight, serverless local database. Unlike Pinecone, it requires no external API keys, network configurations, or cloud service subscriptions. Running ChromaDB locally eliminates latency and supports offline clinical processing, maintaining complete data privacy for sensitive health reports.

---

## 🚀 Setup Instructions

### 1. Clone & Navigate
```bash
git clone https://github.com/saad1032/ai-research-assistant.git
cd medical-report-analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory by copying the template:
```bash
cp .env.example .env
```
Open `.env` and fill in your OpenAI credential:
```env
OPENAI_API_KEY=your_actual_openai_key
```

### 4. Run the FastAPI Backend
Start the backend server on port `8000`:
```bash
uvicorn app.main:app --reload
```

### 5. Run the Streamlit Interface
In a new terminal window, start the frontend UI:
```bash
streamlit run app/ui.py
```

---

## 🔮 Future Improvements

*   **Hybrid Retrieval (BM25 + Semantic)**: Combine keyword search with dense vector embeddings to ensure precise matches on highly specific clinical acronyms and values.
*   **Multi-Modal OCR Ingestion**: Incorporate Tesseract or LayoutLM to support image-based reports, scanned laboratory receipts, and handwritten doctor notes.
*   **Local Clinical Models**: Swap OpenAI for offline clinical open-source LLMs (like Med-Alpaca or ClinicalGPT) to fully protect patient health information (PHI) without internet dependency.
