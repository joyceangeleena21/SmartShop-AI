"""
SmartShop AI - Premium Modern Dark UI Styles
Ensures high contrast, crystal-clear readability for all text, headings,
inactive/active tabs, cards, pills, buttons, tables, and explainability boxes.
"""

CUSTOM_CSS = """
<style>
/* Import Modern Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Reset & High Contrast Typography */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: radial-gradient(circle at 50% 0%, #172036 0%, #0b0f19 75%, #070a12 100%) !important;
    background-attachment: fixed !important;
    color: #F8FAFC !important;
}

/* Headings: Crisp, high-contrast white */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}

/* Paragraphs, labels, and generic text */
p, label, .stMarkdown, span {
    color: #E2E8F0;
}

strong, b {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Streamlit Captions: readable silver rather than dim gray */
.stCaption, [data-testid="stCaptionContainer"], small {
    color: #CBD5E1 !important;
    font-size: 0.88rem !important;
}

/* Top Hero Header */
.hero-container {
    text-align: center;
    padding: 1.8rem 1rem 1.4rem 1rem;
    margin-bottom: 1.5rem;
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #E2E8F0 !important;
    font-weight: 450;
    max-width: 750px;
    margin: 0 auto 1rem auto;
    line-height: 1.5;
}

.hero-pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.5);
    color: #E0E7FF !important;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 10px #10B981;
    display: inline-block;
}

/* =========================================================================
   STREAMLIT TABS: HIGH CONTRAST & READABLE (ACTIVE & INACTIVE)
   ========================================================================= */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: rgba(17, 24, 39, 0.85) !important;
    padding: 8px !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
}

/* Inactive tabs: clearly readable light silver, NOT dark/black */
.stTabs [data-baseweb="tab"] {
    height: 46px !important;
    padding: 0 22px !important;
    background-color: rgba(30, 41, 59, 0.6) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #FFFFFF !important;
    background-color: rgba(99, 102, 241, 0.25) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
}

/* Active selected tab: glowing indigo accent */
.stTabs [aria-selected="true"] {
    background-color: rgba(99, 102, 241, 0.45) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(129, 140, 248, 0.8) !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.3) !important;
}

.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span {
    color: inherit !important;
    font-weight: inherit !important;
}

/* =========================================================================
   SIDEBAR STYLING & HIGH CONTRAST
   ========================================================================= */
[data-testid="stSidebar"] {
    background-color: #0d1322 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] li {
    color: #E2E8F0 !important;
}

[data-testid="stSidebar"] [data-testid="stSlider"] div {
    color: #F1F5F9 !important;
}

[data-testid="stSidebar"] code {
    background: rgba(99, 102, 241, 0.2) !important;
    color: #C7D2FE !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
}

/* =========================================================================
   PRODUCT CARDS & SCORE COMPONENTS
   ========================================================================= */
.product-card {
    background: rgba(26, 34, 53, 0.9);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1.3rem;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.product-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.6);
    box-shadow: 0 12px 30px -10px rgba(99, 102, 241, 0.35);
}

.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 36px;
    height: 36px;
    padding: 0 8px;
    border-radius: 10px;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: #FFFFFF !important;
    font-weight: 800;
    font-size: 0.95rem;
}

.product-title {
    font-size: 1.18rem;
    font-weight: 700;
    color: #FFFFFF !important;
    line-height: 1.4;
    margin-top: 4px;
    margin-bottom: 4px;
}

.brand-pill {
    display: inline-block;
    padding: 3px 10px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #F1F5F9 !important;
    margin-right: 6px;
}

.price-tag {
    font-size: 1.5rem;
    font-weight: 800;
    color: #34D399 !important;
    letter-spacing: -0.02em;
}

.original-price-tag {
    font-size: 0.95rem;
    color: #94A3B8 !important;
    text-decoration: line-through;
    margin-left: 8px;
}

.discount-tag {
    display: inline-block;
    background: rgba(239, 68, 68, 0.25);
    border: 1px solid rgba(239, 68, 68, 0.45);
    color: #FFA4A4 !important;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    margin-left: 6px;
}

/* Score Pills */
.score-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 4px;
}

.score-chip-composite {
    background: rgba(99, 102, 241, 0.3);
    border: 1px solid rgba(99, 102, 241, 0.7);
    color: #E0E7FF !important;
}

.score-chip-semantic {
    background: rgba(14, 165, 233, 0.3);
    border: 1px solid rgba(14, 165, 233, 0.7);
    color: #BAE6FD !important;
}

.score-chip-budget {
    background: rgba(16, 185, 129, 0.3);
    border: 1px solid rgba(16, 185, 129, 0.7);
    color: #A7F3D0 !important;
}

.score-chip-rating {
    background: rgba(245, 158, 11, 0.3);
    border: 1px solid rgba(245, 158, 11, 0.7);
    color: #FDE68A !important;
}

.score-chip-spec {
    background: rgba(168, 85, 247, 0.3);
    border: 1px solid rgba(168, 85, 247, 0.7);
    color: #F3E8FF !important;
}

/* Specifications Grid */
.spec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px;
    margin-bottom: 10px;
    background: rgba(15, 23, 42, 0.6);
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.spec-item {
    font-size: 0.85rem;
    color: #CBD5E1 !important;
    line-height: 1.45;
}

.spec-label {
    color: #94A3B8 !important;
    font-weight: 600;
}

.spec-value {
    color: #FFFFFF !important;
    font-weight: 500;
}

/* Explainability Box */
.explain-box {
    background: rgba(15, 23, 42, 0.9);
    border-left: 4px solid #818CF8;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.4rem;
    margin-top: 0.9rem;
    font-size: 0.92rem;
    line-height: 1.55;
    color: #F8FAFC !important;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.highlight-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 0.86rem;
    color: #6EE7B7 !important;
    margin-top: 5px;
    line-height: 1.4;
}

.tradeoff-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 0.86rem;
    color: #FCA5A5 !important;
    margin-top: 5px;
    line-height: 1.4;
}

/* KPI metric cards */
.kpi-card {
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
}

.kpi-value {
    font-size: 1.4rem;
    font-weight: 800;
    color: #60A5FA !important;
}

.kpi-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #CBD5E1 !important;
    margin-top: 4px;
}

/* Buttons */
.stButton > button {
    background: rgba(30, 41, 59, 0.9) !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: rgba(99, 102, 241, 0.35) !important;
    border-color: rgba(99, 102, 241, 0.7) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4338CA, #6D28D9) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
}

/* Inputs & Select boxes */
.stTextInput input {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
}

.stTextInput input:focus {
    border-color: #818CF8 !important;
    box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.3) !important;
}

.stTextInput input::placeholder {
    color: #94A3B8 !important;
}

/* BaseWeb Selects, Menus, Options, and Tags */
div[data-baseweb="select"] {
    background-color: #1E293B !important;
}

div[data-baseweb="select"] div {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
}

div[data-baseweb="select"] span {
    color: #FFFFFF !important;
}

div[data-baseweb="select"] svg {
    fill: #CBD5E1 !important;
}

ul[role="listbox"] {
    background-color: #1E293B !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
}

li[role="option"] {
    color: #F8FAFC !important;
    background-color: #1E293B !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: rgba(99, 102, 241, 0.4) !important;
    color: #FFFFFF !important;
}

[data-baseweb="tag"] {
    background-color: rgba(99, 102, 241, 0.3) !important;
    border: 1px solid rgba(99, 102, 241, 0.6) !important;
}

[data-baseweb="tag"] span {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

/* Tables and DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    background-color: #0f172a !important;
}

/* Hide default Streamlit clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
