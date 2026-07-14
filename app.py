import os
import streamlit as st
from backend import APP_NAME, EstateGPTAgent, mongo_store

st.set_page_config(page_title=APP_NAME, page_icon="🏠", layout="wide")

# ================= UI Styling =================
st.markdown(
    """
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .main-header {
        font-size: 2.8rem;
        color: #00D4B1;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2.5rem;
        font-size: 1.1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #555555;
        color: #FAFAFA;
        border: none;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #777777;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #393946;
        border-radius: 8px;
        padding: 0.8rem;
    }
    .stTextInput>div>div>input:focus {
        border-color: #00D4B1;
        box-shadow: 0 0 0 2px rgba(0, 212, 177, 0.2);
    }
    .css-1d391kg, .css-1d391kg>div {
        background-color: #0E1117 !important;
        border-right: 1px solid #262730;
    }
    .css-1d391kg h1,h2,h3,h4,h5,h6,p,label {
        color: #FAFAFA !important;
    }
    .stProgress > div > div > div > div {
        background-color: #00D4B1;
    }
    .streamlit-expanderHeader {
        background-color: #262730;
        color: #FAFAFA;
        border-radius: 8px;
        font-weight: 600;
    }
    .streamlit-expanderContent {
        background-color: #1A1D25;
        border-radius: 0 0 8px 8px;
    }
    .card {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #00D4B1;
    }
    .stRadio > div {
        background-color: #262730;
        padding: 1rem;
        border-radius: 8px;
    }
    label {
        font-weight: 600 !important;
        margin-bottom: 0.5rem;
        display: block;
        color: #CCC !important;
    }
    .main-title {
        font-size: 40px;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    .main-title img {
        height: 50px;
        margin-left: 20px;
        vertical-align: middle;
    }
    .subtitle {
        font-size: 20px;
        color: #AAAAAA;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="header">
    <div class="main-title">
        Real Estate With
        <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*QR3Jl4jUu326U2p2.png" alt="ScrapeGraphAI Logo">
        <span style="margin-left:40px;">&</span>
        <img src="https://raw.githubusercontent.com/lobehub/lobe-icons/refs/heads/master/packages/static-png/dark/langchain-color.png" alt="Langchain Logo">
    </div>
    <div class="subtitle">Intelligent Real Estate Matchmaking Engine</div>
    <br>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.image("./assets/Groq.svg", width=150)
    st.markdown("---")

    groq_key = st.text_input(
        "Enter your Groq API key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
    )
    smartscrape_key = st.text_input(
        "Smartscrape Key", value=os.getenv("SCRAPEGRAPH_API_KEY", ""), type="password"
    )

    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state["GROQ_API_KEY"] = groq_key
        st.session_state["SCRAPEGRAPH_API_KEY"] = smartscrape_key
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ["SCRAPEGRAPH_API_KEY"] = smartscrape_key
        st.success("✅ Credentials saved")
        if "agent" in st.session_state:
            del st.session_state["agent"]
        st.rerun()
        
    if groq_key or smartscrape_key:
        st.caption("Credentials loaded for active session.")
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        st.caption("🔍 LangSmith tracing active")

    st.markdown("---")
    st.markdown("### Recents")
    history = mongo_store.get_search_history(limit=10)
    
    clicked_history_query = None
    if not history:
        st.caption("No search history yet.")
    else:
        for i, h in enumerate(history):
            short_h = h[:35] + "..." if len(h) > 35 else h
            if st.button(f"💬 {short_h}", key=f"hist_btn_{i}", help=h):
                clicked_history_query = h

# =========================================================
# Main Search App
# =========================================================

query = st.chat_input("Describe the property you want")

if clicked_history_query:
    query = clicked_history_query

if query:
    with st.spinner("Searching, ranking, and enriching properties..."):
        agent    = EstateGPTAgent()
        response = agent.invoke(query)

    st.chat_message("user").write(query)
    
    with st.chat_message("assistant"):
        st.subheader("Recommendation")
        st.write(response.answer)

    if response.recommended_properties:
        st.subheader("Top Properties")
        for prop in response.recommended_properties:
            with st.container(border=True):
                st.markdown(f"### {prop.title}")
                st.write(f"📍 **{prop.location.locality or prop.location.city or 'Unknown location'}**")
                st.write(f"💰 **{prop.currency} {prop.price or 'N/A'}**")

                # Show up to 3 images side-by-side
                if prop.images:
                    cols = st.columns(3)
                    for i, img in enumerate(prop.images[:3]):
                        with cols[i]:
                            st.image(img.url, use_container_width=True)

                # Full details in an expander
                with st.expander("View full property details"):
                    st.write(f"**Property Type:** {prop.property_type or 'N/A'}")
                    st.write(f"**Area:** {prop.area_sqft or 'N/A'} sqft")
                    st.write(f"**Configuration:** {prop.bedrooms or 'N/A'} BHK, {prop.bathrooms or 'N/A'} Bath")

                    if prop.furnished_status:
                        st.write(f"**Furnishing:** {prop.furnished_status}")
                    if prop.builder:
                        st.write(f"**Builder:** {prop.builder}")
                    if prop.amenities:
                        st.write(f"**Amenities:** {', '.join(prop.amenities)}")

                    st.write(f"**AI Match Score:** {prop.rerank_score:.2f}")
                    st.write(f"**Investment Score:** {prop.investment_score:.1f}")

                    if prop.description:
                        st.write("**Description:**")
                        st.write(prop.description)

                    st.markdown(f"[🔗 Go to original listing]({prop.url})")