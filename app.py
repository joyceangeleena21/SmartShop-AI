"""
SmartShop AI — LLM-Powered Product Recommendation System
Main Streamlit Application Entry Point
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# App imports
from config.config import (
    DEFAULT_WEIGHTS,
    DEFAULT_TOP_K,
    CURRENCY_SYMBOL,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
)
from data.dataset_loader import get_dataset_loader
from models.embedding_engine import get_embedding_engine
from models.intent_extractor import get_intent_extractor, UserIntent
from recommendation.ranker import get_recommendation_engine
from recommendation.explainer import get_recommendation_explainer
from comparison.comparator import get_product_comparator
from ui.styles import CUSTOM_CSS
from ui.components import (
    render_header,
    render_intent_card,
    render_product_card,
    render_radar_chart,
    render_html,
)

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartShop AI — LLM Product Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
render_html(CUSTOM_CSS)

# Initialize singletons
loader = get_dataset_loader()
catalog_df = loader.load()
ranker_engine = get_recommendation_engine()
comparator = get_product_comparator()

# -----------------------------------------------------------------------------
# 2. SIDEBAR HYPERPARAMETERS & SYSTEM CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    st.caption("Customize hybrid ranking factors & thresholds")

    # Ranking Weights
    st.markdown("#### Ranking Weights (Composite Formula)")
    w_sem = st.slider("Semantic Relevance (w_sem)", 0.0, 1.0, DEFAULT_WEIGHTS["semantic"], 0.05)
    w_bud = st.slider("Budget Compatibility (w_budget)", 0.0, 1.0, DEFAULT_WEIGHTS["budget"], 0.05)
    w_rat = st.slider("Rating Credibility (w_rating)", 0.0, 1.0, DEFAULT_WEIGHTS["rating"], 0.05)
    w_feat = st.slider("Feature & Persona Fit (w_feat)", 0.0, 1.0, DEFAULT_WEIGHTS["feature"], 0.05)

    custom_weights = {
        "semantic": w_sem,
        "budget": w_bud,
        "rating": w_rat,
        "feature": w_feat,
    }

    # Filter toggles
    st.markdown("---")
    st.markdown("#### Search Constraints")
    strict_category = st.checkbox("Strict Category Filter", value=True, help="Restrict candidates to the category detected in your query")
    strict_budget = st.checkbox("Strict Budget Cap", value=False, help="Filter out any products strictly exceeding the extracted budget")
    top_k = st.slider("Number of Recommendations (Top-K)", min_value=3, max_value=12, value=5, step=1)

    # Catalog & Engine Diagnostics
    st.markdown("---")
    st.markdown("#### System Diagnostics")
    stats = loader.get_catalog_stats()
    st.markdown(f"""
    - **Total Products:** {stats['total_products']} items
    - **Categories:** {stats['categories_count']}
    - **Dense Model:** `{EMBEDDING_MODEL_NAME.split('/')[-1]}`
    - **Vector Dim:** {EMBEDDING_DIM}-d dense
    - **Device:** Local CPU (Zero Cost)
    - **Embedding Cache:** Active
    """)


# -----------------------------------------------------------------------------
# 3. TOP HERO HEADER
# -----------------------------------------------------------------------------
render_header()

# Initialize session state for search query
if "search_query" not in st.session_state:
    st.session_state["search_query"] = "I need a laptop under 70000 for Python, machine learning and college work."

# -----------------------------------------------------------------------------
# 4. TABS NAVIGATION
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "🔎 Smart Recommender",
    "⚖️ Product Comparison Matrix",
    "📦 Product Catalog Explorer",
])

# =============================================================================
# TAB 1: SMART RECOMMENDER
# =============================================================================
with tabs[0]:
    st.markdown("#### 💬 Ask SmartShop AI in Natural Language")
    st.caption("Type your requirements or click any quick-start sample query below:")

    # Quick Preset Query Buttons
    presets = [
        ("💻 ML & Python Laptop under 70k", "I need a laptop under 70000 for Python, machine learning and college work."),
        ("🎮 RTX 3050 Gaming Laptop < 75k", "Gaming laptop with RTX 3050 and 16GB RAM under 75k"),
        ("⚡ Lightweight Ultrabook under 60k", "Lightweight ultrabook under 60000 for long battery life and coding"),
        ("🎓 Budget Student Laptop < 40k", "Budget student laptop under 40000 for online classes and MS Office"),
        ("📱 120Hz Phone under 25k", "Best smartphone under 25000 with 120Hz AMOLED display and good camera"),
        ("🎧 Noise Cancelling Headphones < 10k", "Sony or Bose headphones under 10k with noise cancellation for study"),
        ("🍏 MacBook for iOS Dev", "MacBook for iOS development and Swift under 1.3 lakh"),
    ]

    cols_preset = st.columns(len(presets))
    for i, (label, text) in enumerate(presets):
        if cols_preset[i].button(label, key=f"preset_{i}", use_container_width=True):
            st.session_state["search_query"] = text

    # Main Search Input
    query_input = st.text_input(
        "Search Query",
        value=st.session_state["search_query"],
        placeholder="e.g. I need a laptop under 70000 for Python, machine learning and college work.",
        label_visibility="collapsed",
    )

    search_col1, search_col2 = st.columns([1, 5])
    with search_col1:
        search_clicked = st.button("🚀 Find Recommendations", type="primary", use_container_width=True)

    if query_input:
        with st.spinner("Analyzing requirements, embedding query, and ranking candidates..."):
            results, detected_intent = ranker_engine.recommend(
                query=query_input,
                top_k=top_k,
                weights=custom_weights,
                strict_budget=strict_budget,
                strict_category=strict_category,
            )

        # Render Extracted Intent KPI Card
        st.markdown("##### 🎯 Extracted Requirement Blueprint")
        render_intent_card(detected_intent)

        st.markdown(f"##### 🏆 Top {len(results)} Ranked Recommendations for You")

        if not results:
            st.warning("No products in the catalog satisfied all strict filters. Try unchecking 'Strict Budget' or modifying your query.")
        else:
            for rank_idx, rec in enumerate(results, 1):
                render_product_card(rec, detected_intent, rank_idx)


# =============================================================================
# TAB 2: PRODUCT COMPARISON MATRIX
# =============================================================================
with tabs[1]:
    st.markdown("#### ⚖️ Side-by-Side Technical Comparison Matrix")
    st.caption("Select 2 to 4 products from the catalog to compare hardware specifications, prices, customer ratings, and radar metrics.")

    # Product Selector
    catalog_prods = catalog_df.to_dict(orient="records")
    product_options = {
        f"[{p['category']}] {p['brand']} - {p['title'][:55]}... ({p['price_formatted']})": p['product_id']
        for p in catalog_prods
    }

    # Default options: Acer Nitro V vs Lenovo LOQ
    default_ids = ["LAP_001", "LAP_002", "LAP_007"]
    default_keys = [k for k, v in product_options.items() if v in default_ids]

    selected_keys = st.multiselect(
        "Choose Products to Compare:",
        options=list(product_options.keys()),
        default=default_keys[:3],
        max_selections=4,
    )

    if len(selected_keys) < 2:
        st.info("Please select at least 2 products to generate the comparison matrix and radar chart.")
    else:
        selected_ids = [product_options[k] for k in selected_keys]
        selected_prods = [loader.get_product_by_id(pid) for pid in selected_ids if loader.get_product_by_id(pid)]

        # Category Winners Banner
        winners = comparator.find_category_winners(selected_prods)
        win_cols = st.columns(len(winners))
        for idx, (title, winner_desc) in enumerate(winners.items()):
            with win_cols[idx]:
                render_html(
                    f"""
                    <div class="kpi-card" style="padding: 0.8rem;">
                        <div style="font-size: 0.82rem; font-weight: 700; color: #10B981; margin-bottom: 4px;">🏆 {title}</div>
                        <div style="font-size: 0.85rem; color: #E2E8F0;">{winner_desc}</div>
                    </div>
                    """
                )

        st.markdown("---")

        # Side-by-side Table & Radar Chart in 2 Columns
        col_table, col_radar = st.columns([3, 2])

        with col_table:
            st.markdown("##### 📋 Specifications Breakdown")
            comp_df = comparator.build_comparison_table(selected_prods)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        with col_radar:
            st.markdown("##### 🎯 Multi-Dimensional Radar Comparison")
            radar_data = comparator.compute_radar_metrics(selected_prods)
            fig_radar = render_radar_chart(radar_data)
            st.plotly_chart(fig_radar, use_container_width=True)


# =============================================================================
# TAB 3: PRODUCT CATALOG EXPLORER
# =============================================================================
with tabs[2]:
    st.markdown("#### 📦 Product Catalog Explorer")
    st.caption("Browse, search, and filter the underlying e-commerce dataset.")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        cat_filter = st.selectbox("Category", ["All"] + loader.get_all_categories())
    with c2:
        brand_filter = st.selectbox("Brand", ["All"] + loader.get_all_brands())
    with c3:
        search_kw = st.text_input("Filter by Keyword in Title or Specs", "")

    filtered_df = catalog_df.copy()
    if cat_filter != "All":
        filtered_df = filtered_df[filtered_df["category"] == cat_filter]
    if brand_filter != "All":
        filtered_df = filtered_df[filtered_df["brand"] == brand_filter]
    if search_kw:
        filtered_df = filtered_df[
            filtered_df["semantic_representation"].str.contains(search_kw, case=False, na=False)
        ]

    st.markdown(f"Showing **{len(filtered_df)}** of **{len(catalog_df)}** products:")
    display_cols = ["product_id", "title", "brand", "category", "price_formatted", "rating", "review_count", "processor", "gpu", "ram", "storage"]
    st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)
