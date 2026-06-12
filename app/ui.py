"""
Streamlit User Interface for the Medical Report Analyzer.
Provides a front-end UI for uploading reports, configuring settings, and interacting with the RAG agent.
"""

import streamlit as st
import requests

# Backend URL configuration
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Medical Report Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS injection
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 1.25rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.25rem;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #1F2937;
    }
    .answer-box-flagged {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1.25rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.25rem;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #1F2937;
    }
    .score-badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.85em;
        font-weight: 700;
        line-height: 1;
        color: #fff;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
    }
    .badge-success {
        background-color: #10B981;
    }
    .badge-danger {
        background-color: #EF4444;
    }
    .compare-card {
        background: #F9FAFB;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("# 🩺 Medical Report Analyzer")
st.sidebar.write("Analyze and interpret medical lab reports using semantic RAG.")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📤 Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF lab report", 
    type=["pdf"], 
    help="Upload medical report PDF to index into the vector store."
)

if uploaded_file is not None:
    if st.sidebar.button("⚙️ Ingest & Process Report", use_container_width=True):
        with st.sidebar.spinner("Uploading and indexing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                res = requests.post(f"{BACKEND_URL}/upload", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.sidebar.success(
                        f"**Ingestion Success!**\n"
                        f"- Semantic: {data['semantic_chunks_count']} chunks\n"
                        f"- Fixed: {data['fixed_chunks_count']} chunks"
                    )
                else:
                    error_detail = res.json().get("detail", "Unknown server error")
                    st.sidebar.error(f"Ingestion failed: {error_detail}")
            except Exception as e:
                st.sidebar.error(f"Could not connect to backend server: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ RAG Settings")
strategy_option = st.sidebar.radio(
    "Select Chunking Strategy",
    options=["Semantic", "Fixed"],
    index=0,
    help="Semantic respects paragraph/sentence structure. Fixed splits text into exact character offsets."
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with LangChain + ChromaDB + Streamlit")


# ----------------- MAIN LAYOUT -----------------
st.markdown('<div class="main-header">🩺 Medical Report RAG Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a PDF report in the sidebar and query medical findings using custom chunking strategies.</div>', 
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["🔍 Ask Report", "📊 Strategy Comparison"])

# --- Tab 1: Ask Report ---
with tab1:
    st.write("### Query Lab Findings")
    query = st.text_input(
        "Enter your question about the lab report:",
        placeholder="e.g., What are my Hemoglobin levels, and are they in the normal range?",
        key="main_query"
    )
    
    if st.button("Submit Query", type="primary", use_container_width=False):
        if not query.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Searching vector store and generating response..."):
                payload = {
                    "query": query,
                    "strategy": strategy_option.lower()
                }
                try:
                    res = requests.post(f"{BACKEND_URL}/ask", json=payload)
                    if res.status_code == 200:
                        result = res.json()
                        answer = result["answer"]
                        confidence = result["confidence_score"]
                        is_flagged = result["is_flagged"]
                        source_chunks = result["source_chunks"]
                        
                        # Answer Output
                        st.write("#### Response")
                        box_class = "answer-box-flagged" if is_flagged else "answer-box"
                        st.markdown(f'<div class="{box_class}">{answer}</div>', unsafe_allow_html=True)
                        
                        # Confidence Metric
                        st.write("#### Confidence Score (Hallucination Guard)")
                        badge_color = "badge-danger" if is_flagged else "badge-success"
                        badge_label = "Low Confidence (Flagged)" if is_flagged else "High Confidence"
                        st.markdown(
                            f'<span class="score-badge {badge_color}">{badge_label}: {confidence:.4f}</span>', 
                            unsafe_allow_html=True
                        )
                        
                        # Progress bar representation
                        progress_color = "#EF4444" if is_flagged else "#10B981"
                        st.markdown(f"""
                        <div style="background-color: #E5E7EB; border-radius: 5px; height: 10px; width: 100%; margin-bottom: 2rem;">
                            <div style="background-color: {progress_color}; width: {confidence * 100}%; height: 100%; border-radius: 5px;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Retrieved context details
                        st.write("#### Reference Sources Used")
                        for idx, chunk in enumerate(source_chunks):
                            score = chunk["similarity_score"]
                            content = chunk["content"]
                            meta = chunk["metadata"]
                            
                            with st.expander(
                                f"Chunk {idx + 1} | Similarity: {score:.4f} "
                                f"(Index: {meta['chunk_index']}, Source: {meta['source']})"
                            ):
                                st.code(content, language="text")
                                
                    elif res.status_code == 404:
                        st.warning("⚠️ No database found. Please upload and process a medical report PDF first.")
                    else:
                        error_detail = res.json().get("detail", "Server error occurred.")
                        st.error(f"Error ({res.status_code}): {error_detail}")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")

# --- Tab 2: Strategy Comparison ---
with tab2:
    st.write("### Compare Chunking Strategies Side-by-Side")
    st.write(
        "Observe how the different text chunking algorithms select and score context from the same document."
    )
    
    compare_query = st.text_input(
        "Enter search terms or query to compare:",
        placeholder="e.g., hemoglobin or lymphocytes",
        key="compare_query"
    )
    
    if st.button("Run Comparison", type="secondary"):
        if not compare_query.strip():
            st.warning("Please enter search terms first.")
        else:
            with st.spinner("Fetching matching chunks for comparison..."):
                try:
                    res = requests.get(f"{BACKEND_URL}/compare", params={"query": compare_query})
                    if res.status_code == 200:
                        data = res.json()
                        semantic_results = data.get("semantic", [])
                        fixed_results = data.get("fixed", [])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🧬 Semantic Chunking")
                            st.caption("RecursiveCharacterTextSplitter (chunk_size=512, overlap=64)")
                            st.markdown("---")
                            
                            if not semantic_results:
                                st.info("No matching chunks found in semantic vector space.")
                            else:
                                # Show top 3
                                for i, chunk in enumerate(semantic_results[:3]):
                                    st.markdown(f"""
                                    <div class="compare-card">
                                        <strong>Rank {i+1} | Cosine Similarity: {chunk['similarity_score']:.4f}</strong><br/>
                                        <small>Source: {chunk['metadata']['source']} | Index: {chunk['metadata']['chunk_index']}</small>
                                        <hr style="margin: 0.5rem 0; border-top: 1px solid #E5E7EB;"/>
                                        <p style="font-size: 0.92rem; color: #374151; line-height: 1.5; white-space: pre-wrap;">{chunk['content']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                        with col2:
                            st.subheader("📏 Fixed-Size Chunking")
                            st.caption("Fixed offset split (chunk_size=512, overlap=0)")
                            st.markdown("---")
                            
                            if not fixed_results:
                                st.info("No matching chunks found in fixed vector space.")
                            else:
                                # Show top 3
                                for i, chunk in enumerate(fixed_results[:3]):
                                    st.markdown(f"""
                                    <div class="compare-card">
                                        <strong>Rank {i+1} | Cosine Similarity: {chunk['similarity_score']:.4f}</strong><br/>
                                        <small>Source: {chunk['metadata']['source']} | Index: {chunk['metadata']['chunk_index']}</small>
                                        <hr style="margin: 0.5rem 0; border-top: 1px solid #E5E7EB;"/>
                                        <p style="font-size: 0.92rem; color: #374151; line-height: 1.5; white-space: pre-wrap;">{chunk['content']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                    elif res.status_code == 404:
                        st.warning("⚠️ No database found. Please upload and process a medical report PDF first.")
                    else:
                        error_detail = res.json().get("detail", "Server error occurred.")
                        st.error(f"Error ({res.status_code}): {error_detail}")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")

