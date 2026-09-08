"""
SmartShop AI - Reusable UI Components & Visualizations
Renders rich product recommendation cards, explainability drawers,
interactive Plotly radar charts, and evaluation metric dashboards.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import plotly.graph_objects as go
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.intent_extractor import UserIntent
from recommendation.ranker import RecommendationResult
from recommendation.explainer import get_recommendation_explainer
from config.config import CURRENCY_SYMBOL


import textwrap

def render_html(html_code: str):
    """
    Renders HTML cleanly using st.html (Streamlit 1.33+) to avoid CommonMark
    code-block indentation issues, with fallback to dedented st.markdown.
    """
    clean_html = textwrap.dedent(html_code).strip()
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)


def render_header():
    """Renders the top hero section with status pills."""
    render_html(
        """
        <div class="hero-container">
            <div class="hero-title">SmartShop AI</div>
            <div class="hero-subtitle">
                Next-Gen LLM & Embedding-Powered Product Recommendation Engine with Multi-Factor Hybrid Ranking & Explainable AI
            </div>
            <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                <span class="hero-pill-badge">
                    <span class="status-dot"></span> all-MiniLM-L6-v2 (384-d Dense Vectors)
                </span>
                <span class="hero-pill-badge">
                    Multi-Factor Ranking (Semantic + Budget + Rating + Specs)
                </span>
                <span class="hero-pill-badge">
                    100% Free / Open-Source (Runs Offline on CPU)
                </span>
            </div>
        </div>
        """
    )


def render_intent_card(intent: UserIntent):
    """Renders structured feedback on what the system extracted from the user's natural language query."""
    cols = st.columns(4)
    with cols[0]:
        cat_text = intent.category if intent.category else "All Categories"
        render_html(
            f"""
            <div class="kpi-card">
                <div class="kpi-value" style="font-size: 1.3rem;">{cat_text}</div>
                <div class="kpi-label">Detected Category</div>
            </div>
            """
        )

    with cols[1]:
        if intent.budget_max:
            bud_text = f"{CURRENCY_SYMBOL}{intent.budget_max:,}"
        else:
            bud_text = "Flexible"
        render_html(
            f"""
            <div class="kpi-card">
                <div class="kpi-value" style="font-size: 1.3rem; color: #10B981;">{bud_text}</div>
                <div class="kpi-label">Max Budget Cap</div>
            </div>
            """
        )

    with cols[2]:
        ucs = [u.replace("_", " ").title() for u in intent.use_cases]
        uc_text = ", ".join(ucs) if ucs else "General Purpose"
        render_html(
            f"""
            <div class="kpi-card">
                <div class="kpi-value" style="font-size: 1.1rem; line-height: 1.2;">{uc_text}</div>
                <div class="kpi-label">Primary Intent & Use-Case</div>
            </div>
            """
        )

    with cols[3]:
        specs = [f"{k.upper()}: {v}" for k, v in intent.specs.items()]
        spec_text = ", ".join(specs) if specs else "Standard Spec"
        render_html(
            f"""
            <div class="kpi-card">
                <div class="kpi-value" style="font-size: 1.1rem; line-height: 1.2;">{spec_text}</div>
                <div class="kpi-label">Extracted Key Specs</div>
            </div>
            """
        )


def render_product_card(result: RecommendationResult, intent: UserIntent, rank: int):
    """Renders a comprehensive, interactive product recommendation card with explainability."""
    p = result.product
    explainer = get_recommendation_explainer()
    dossier = explainer.explain(result, intent)

    highlights_html = "".join([
        f'<div class="highlight-item"><span>✔</span> {h}</div>' for h in dossier["highlights"]
    ])
    tradeoffs_html = "".join([
        f'<div class="tradeoff-item"><span>⚠</span> {t}</div>' for t in dossier["trade_offs"]
    ])

    card_html = f"""
    <div class="product-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span class="rank-badge">#{rank}</span>
                <div>
                    <span class="brand-pill">{p.get('brand', '')}</span>
                    <span class="brand-pill">{p.get('category', '')}</span>
                    <span class="brand-pill" style="background: rgba(99, 102, 241, 0.2); color: #A5B4FC;">{dossier['headline']}</span>
                    <div class="product-title">{p.get('title', '')}</div>
                </div>
            </div>
            <div style="text-align: right; min-width: 150px;">
                <div>
                    <span class="price-tag">{p.get('price_formatted', '')}</span>
                    <span class="original-price-tag">{CURRENCY_SYMBOL}{p.get('original_price', 0):,}</span>
                </div>
                <div>
                    <span class="discount-tag">{p.get('discount_pct', 0)}% OFF</span>
                    <span style="color: #94A3B8; font-size: 0.8rem; font-weight: 600; margin-left: 6px;">
                        {result.budget_status}
                    </span>
                </div>
                <div style="color: #FBBF24; font-size: 0.85rem; font-weight: 600; margin-top: 4px;">
                    {'★' * int(round(p.get('rating', 4.0)))} {p.get('rating', 4.0)}★ ({p.get('review_count', 0):,} reviews)
                </div>
            </div>
        </div>

        <div style="margin-bottom: 12px;">
            <span class="score-chip score-chip-composite">Composite Fit: {dossier['score_breakdown']['composite_pct']}%</span>
            <span class="score-chip score-chip-semantic">Semantic Match: {dossier['score_breakdown']['semantic_match_pct']}%</span>
            <span class="score-chip score-chip-budget">Budget Fit: {dossier['score_breakdown']['budget_fit_pct']}%</span>
            <span class="score-chip score-chip-rating">Rating Trust: {dossier['score_breakdown']['rating_credibility_pct']}%</span>
            <span class="score-chip score-chip-spec">Spec Match: {dossier['score_breakdown']['spec_alignment_pct']}%</span>
        </div>

        <div class="spec-grid">
            <div class="spec-item"><span class="spec-label">Processor:</span> <span class="spec-value">{p.get('processor', 'N/A')}</span></div>
            <div class="spec-item"><span class="spec-label">Graphics (GPU):</span> <span class="spec-value">{p.get('gpu', 'N/A')}</span></div>
            <div class="spec-item"><span class="spec-label">Memory (RAM):</span> <span class="spec-value">{p.get('ram', 'N/A')}</span></div>
            <div class="spec-item"><span class="spec-label">Storage:</span> <span class="spec-value">{p.get('storage', 'N/A')}</span></div>
            <div class="spec-item"><span class="spec-label">Display:</span> <span class="spec-value">{p.get('display', 'N/A')}</span></div>
            <div class="spec-item"><span class="spec-label">Battery:</span> <span class="spec-value">{p.get('battery_life', 'N/A')}</span></div>
        </div>

        <div class="explain-box">
            <div style="font-weight: 700; color: #818CF8; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                <span>💡</span> Why SmartShop AI Recommended This:
            </div>
            <div style="color: #F1F5F9; line-height: 1.55;">{dossier['narrative']}</div>

            <div style="margin-top: 10px; font-weight: 600; color: #34D399; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">Key Matching Strengths:</div>
            {highlights_html}

            <div style="margin-top: 10px; font-weight: 600; color: #F87171; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">Trade-offs & Purchase Considerations:</div>
            {tradeoffs_html}
        </div>
    </div>
    """

    with st.container():
        render_html(card_html)


def render_radar_chart(radar_data: List[Dict[str, Any]]) -> go.Figure:
    """Creates a multi-product Plotly radar chart comparing 5 technical dimensions."""
    fig = go.Figure()

    colors = [
        "rgba(99, 102, 241, 0.8)",
        "rgba(16, 185, 129, 0.8)",
        "rgba(244, 63, 94, 0.8)",
        "rgba(245, 158, 11, 0.8)",
    ]
    fill_colors = [
        "rgba(99, 102, 241, 0.2)",
        "rgba(16, 185, 129, 0.2)",
        "rgba(244, 63, 94, 0.2)",
        "rgba(245, 158, 11, 0.2)",
    ]

    for idx, item in enumerate(radar_data):
        metrics = item["metrics"] + [item["metrics"][0]]  # Close the radar loop
        scores = item["scores"] + [item["scores"][0]]

        c = colors[idx % len(colors)]
        fc = fill_colors[idx % len(fill_colors)]

        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=metrics,
            fill='toself',
            name=item["product_name"],
            line=dict(color=c, width=2.5),
            fillcolor=fc,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticks="outside",
                gridcolor="rgba(255, 255, 255, 0.12)",
                color="#94A3B8"
            ),
            angularaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.12)",
                color="#F1F5F9"
            ),
            bgcolor="rgba(15, 23, 42, 0.4)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(
            font=dict(color="#F3F4F6", size=11),
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        height=420
    )

    return fig


def render_ablation_chart(summary_df) -> go.Figure:
    """Renders a grouped bar chart comparing Precision, Recall, MRR, and NDCG across systems."""
    metrics = ["Precision@5", "Recall@5", "MRR", "NDCG@5"]
    fig = go.Figure()

    color_map = {
        "Lexical Keyword Search": "#94A3B8",
        "Pure Semantic Retrieval": "#38BDF8",
        "SmartShop AI (Hybrid)": "#6366F1",
    }

    for _, row in summary_df.iterrows():
        system = row["System Architecture"]
        values = [row[m] for m in metrics]

        fig.add_trace(go.Bar(
            name=system,
            x=metrics,
            y=values,
            marker_color=color_map.get(system, "#6366F1"),
            text=[f"{v:.3f}" for v in values],
            textposition="auto",
        ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(color="#F3F4F6", family="Inter"),
        yaxis=dict(
            range=[0, 1.1],
            gridcolor="rgba(255, 255, 255, 0.08)",
            title="Score (0.0 to 1.0)",
            color="#94A3B8"
        ),
        xaxis=dict(
            color="#F1F5F9",
            title="Ranking Evaluation Metrics"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#F3F4F6")
        ),
        margin=dict(l=30, r=30, t=40, b=30),
        height=380
    )

    return fig
