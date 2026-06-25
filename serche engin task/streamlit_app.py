import sys
import os
import asyncio
import streamlit as st

# Add project root to python path to resolve modules correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app.db.session import init_db
from backend.app.services.retrieval_pipeline import retrieval_pipeline
from backend.app.db.qdrant import qdrant_client, QdrantManager
from backend.app.config import settings

# Set up page configurations
st.set_page_config(
    page_title="AI Dataset Advisor - Intelligent Search",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling overlays
st.markdown("""
<style>
    /* Dark Theme Core Overlay */
    .stApp {
        background: linear-gradient(135deg, #0f0c1b 0%, #15102a 50%, #0c0816 100%);
        color: #e2e8f0;
    }
    
    /* Glassmorphic recommendation container */
    .recommendation-card {
        background: rgba(25, 18, 50, 0.45);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .recommendation-card:hover {
        border-color: rgba(139, 92, 246, 0.6);
        box-shadow: 0 8px 32px 0 rgba(139, 92, 246, 0.15);
        transform: translateY(-2px);
    }
    
    /* Glowing Title Styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa 0%, #ec4899 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    /* Metadata Pills */
    .meta-pill {
        display: inline-block;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: #c084fc;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 8px;
        margin-bottom: 5px;
    }
    .meta-pill-green {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 8px;
        margin-bottom: 5px;
    }
    
    /* Match Gauge */
    .match-gauge {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f472b6;
        float: right;
    }
    
    /* RTL adjustments for Arabic */
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Kufi Arabic', sans-serif;
    }
    
    /* Floating loader bar */
    .loader-stage {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to run async retrieval pipeline
def execute_search_sync(query: str, session_id: str, db_session):
    return asyncio.run(retrieval_pipeline.execute_search(
        raw_query=query,
        session_id=session_id,
        db=db_session
    ))

# Helper to run async db creation
def init_db_sync():
    asyncio.run(init_db())

# Initialize DB on boot
init_db_sync()

# Sidebar configurations
with st.sidebar:
    st.image("https://img.icons8.com/cosmic/128/artificial-intelligence.png", width=70)
    st.markdown("### ⚙️ Advisor Controls")
    
    # Model configuration choice
    model_override = st.selectbox(
        "Active LLM Model",
        [settings.GROQ_MODEL, "llama-3.1-8b-instant"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Database Diagnostics")
    try:
        q_count = qdrant_client.count(collection_name=QdrantManager.COLLECTION_NAME).count
        st.metric("Vector Points Count", q_count)
        st.success("Vector DB Online")
    except Exception as e:
        st.error(f"Vector DB Offline: {e}")
        
    st.markdown("---")
    st.caption("AI Dataset Advisor v1.0.0\nMultilingual Hybrid Retrieval System")

# App Header
col_header_left, col_header_right = st.columns([4, 1])
with col_header_left:
    st.markdown('<div class="main-title">AI Dataset Advisor</div>', unsafe_allow_html=True)
    st.markdown("##### Discover suitable machine learning datasets using Arabic or English queries instantly.")
with col_header_right:
    st.button("🔄 Clear Workspace", on_click=lambda: st.session_state.clear())

# Predefined prompt chips
st.markdown("💬 **Suggested Prompts / نماذج لطلبات البحث:**")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)

with col_p1:
    if st.button("Arabic Sentiment Analysis\nتصنيف مشاعر التغريدات بالعربية"):
        st.session_state.search_query = "أريد داتا صغيرة لتصنيف نصوص المشاعر باللغة العربية"
with col_p2:
    if st.button("USA Housing Prices\nتوقع أسعار المنازل في أمريكا"):
        st.session_state.search_query = "I need a dataset for house price prediction in the USA under 10000 rows"
with col_p3:
    if st.button("Vehicle Detection CV\nرؤية حاسوبية لتتبع السيارات"):
        st.session_state.search_query = "Computer vision dataset for vehicle detection"
with col_p4:
    if st.button("Arabic News Summary\nتلخيص النصوص الإخبارية"):
        st.session_state.search_query = "أريد داتا لتلخيص النصوص الإخبارية باللغة العربية"

# Main Search Input
search_input = st.text_input(
    "Describe the dataset or machine learning task you want to execute:",
    value=st.session_state.get("search_query", ""),
    placeholder="e.g., Arabic tabular dataset for customer churn prediction under 5000 rows...",
    key="main_query_field"
)

# Run Query Search
if search_input:
    # Set search_query session state to enable seamless suggest clicks
    st.session_state.search_query = search_input
    
    st.markdown("---")
    st.markdown("### 🕵️ AI Search Results")
    
    with st.spinner("🤖 Advisor Agent is parsing intent, searching vector spaces, and reranking candidates..."):
        try:
            # We import session_factory on the fly to avoid async lock issues inside streamlit loop
            from backend.app.db.session import async_session_factory
            
            # Establish context and query backend
            async def query_runner():
                async with async_session_factory() as session:
                    res = await retrieval_pipeline.execute_search(
                        raw_query=search_input,
                        session_id="streamlit-session",
                        db=session
                    )
                    return res
                    
            res = asyncio.run(query_runner())
            
            # Render extraction analytics
            intent = res.get("intent", {})
            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            with col_a1:
                st.metric("Detected Language", str(intent.get("detected_language", "Unknown")).upper())
            with col_a2:
                st.metric("Target ML Task", str(intent.get("task_type", "General")).upper())
            with col_a3:
                st.metric("Constraint: Max Rows", str(intent.get("max_rows", "Unlimited")))
            with col_a4:
                st.metric("Modality", str(intent.get("modality", "Tabular")).upper())
                
            st.markdown(f"**Extracted Query Query Vector Translation**: `\"{intent.get('semantic_search_query', '')}\"`")
            
            # Display matched Datasets
            datasets = res.get("datasets", [])
            if not datasets:
                st.warning("⚠️ No datasets matched the specific semantic filters. Try broadening your query terms!")
            else:
                st.markdown(f"Found **{len(datasets)}** matched candidate datasets:")
                
                for idx, ds in enumerate(datasets, 1):
                    # Check RTL logic for arabic datasets
                    is_arabic = ds.get("language") == "arabic"
                    rtl_class = "rtl-text" if is_arabic else ""
                    
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <span class="match-gauge">🎯 Match score: {ds.get('relevance_score', 0)}%</span>
                        <h4 class="{rtl_class}" style="color: #c084fc; margin-bottom: 5px;">{idx}. {ds.get('title')}</h4>
                        <div class="{rtl_class}" style="margin-bottom: 10px;">
                            <span class="meta-pill">📂 Language: {str(ds.get('language')).upper()}</span>
                            <span class="meta-pill">🏷️ Task: {str(ds.get('task_type')).upper()}</span>
                            <span class="meta-pill-green">📊 Rows: {ds.get('row_count') or 'Unknown'}</span>
                            <span class="meta-pill-green">🗂️ Cols: {ds.get('column_count') or 'Unknown'}</span>
                            <span class="meta-pill">⚖️ License: {ds.get('license') or 'Open'}</span>
                        </div>
                        <p class="{rtl_class}" style="color: #94a3b8; font-size: 0.95rem;">{ds.get('description')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Embed AI Insight Expandable Drawer
                    with st.expander(f"💡 AI Advisor Insight / تحليل المستشار الذكي - Dataset {idx}", expanded=(idx==1)):
                        # Match RTL layout for reasoning paragraph if Arabic target
                        reason_text = ds.get('reasoning', '')
                        if is_arabic:
                            st.markdown(f'<div class="rtl-text" style="color: #cbd5e1; line-height: 1.6;">{reason_text}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div style="color: #cbd5e1; line-height: 1.6;">{reason_text}</div>', unsafe_allow_html=True)
                            
                        st.markdown("---")
                        col_url_l, col_url_r = st.columns([1, 4])
                        with col_url_l:
                            st.link_button("🌐 Download Dataset", ds.get("url", "#"), type="primary")
                        with col_url_r:
                            st.caption(f"Source URL: {ds.get('url')}")
                            
        except Exception as e:
            st.error(f"Search Execution crashed: {e}")
            st.exception(e)
            
else:
    st.info("💡 Input a description above or click one of our suggested prompts to see the search advisor in action!")
