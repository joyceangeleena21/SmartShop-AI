# SmartShop AI — AI-Powered Product Recommendation System

SmartShop AI is an end-to-end e-commerce product recommendation application that understands natural-language shopping requirements and recommends suitable products using semantic matching and hybrid ranking.

Instead of searching only by exact keywords, users can describe their requirements in plain English, such as:

> "I need a laptop under ₹70,000 for Python, machine learning and college work."

The system extracts relevant requirements, generates semantic embeddings using a pretrained Sentence Transformer model, and ranks products based on multiple factors such as semantic relevance, budget compatibility, rating, and feature matching.

---

## 🚀 Features

### 1. Natural-Language Product Search

Users can enter requirements in normal language instead of using structured filters.

Example:

```text
I need a laptop under ₹70,000 for Python, machine learning and college work.

The system identifies information such as:

Product category
Budget
RAM requirements
GPU requirements
Operating system
Intended use case
2. Semantic Product Matching

SmartShop AI uses the pretrained Sentence Transformer model:

sentence-transformers/all-MiniLM-L6-v2

to convert user queries and product descriptions into 384-dimensional embeddings.

This allows the system to identify products with similar meanings even when the exact keywords are different.

3. Hybrid Recommendation Ranking

Recommendations are ranked using multiple signals:

Semantic similarity
Budget compatibility
Product rating
Feature/specification matching

The overall recommendation score combines these factors to produce a balanced ranking.

4. Explainable Recommendations

For every recommended product, the application provides an explanation of why the product was selected.

Users can see factors such as:

Semantic match
Budget match
Rating
Specification match
Product strengths and trade-offs
5. Product Comparison

Users can compare multiple products side-by-side based on their technical specifications and other product attributes.

6. Product Catalog Explorer

The application includes a catalog explorer where users can browse the available products and inspect their details.

7. Interactive Streamlit Application

The complete recommendation system is available through an interactive Streamlit web interface.

🧠 How It Works

The recommendation pipeline follows these steps:

User Query
    ↓
Intent & Requirement Extraction
    ↓
Query Embedding
    ↓
Semantic Similarity
    ↓
Hybrid Ranking
    ↓
Top Recommended Products
    ↓
Explanation & Comparison
Step 1 — User Query

The user describes what they are looking for in natural language.

Example:

I need a laptop under ₹70,000 for programming and machine learning.
Step 2 — Requirement Extraction

The system extracts relevant constraints from the query, such as:

Budget
Product category
RAM
GPU
Operating system
Use case
Step 3 — Semantic Embedding

The user query is converted into a numerical vector using:

all-MiniLM-L6-v2

Product descriptions are represented using the same embedding model.

Step 4 — Similarity Calculation

The system compares the query embedding with product embeddings using semantic similarity.

Step 5 — Hybrid Ranking

The semantic score is combined with budget, rating, and specification matching to calculate the final recommendation score.

Step 6 — Explanation

The application presents the recommended products along with the factors contributing to their ranking.

🛠️ Technology Stack
Technology	Purpose
Python	Core development
Pandas	Product data processing
NumPy	Numerical computation
Scikit-learn	Similarity and ML utilities
Sentence Transformers	Semantic embeddings
Streamlit	Web application
Plotly	Interactive visualizations
unittest	Automated testing
📂 Project Structure
SmartShop-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── config/
│   ├── __init__.py
│   └── config.py
│
├── data/
│   ├── __init__.py
│   ├── dataset_generator.py
│   ├── dataset_loader.py
│   ├── metadata.json
│   └── products.csv
│
├── models/
│   ├── __init__.py
│   ├── embedding_engine.py
│   └── intent_extractor.py
│
├── recommendation/
│   ├── __init__.py
│   ├── ranker.py
│   └── explainer.py
│
├── comparison/
│   ├── __init__.py
│   └── comparator.py
│
├── evaluation/
│   ├── __init__.py
│   ├── benchmark_queries.py
│   ├── evaluator.py
│   └── latency_profiler.py
│
├── ui/
│   ├── __init__.py
│   ├── components.py
│   └── styles.py
│
└── tests/
    ├── __init__.py
    ├── test_dataset.py
    ├── test_embedding_engine.py
    ├── test_end_to_end.py
    ├── test_evaluator.py
    ├── test_explainer.py
    ├── test_intent_extractor.py
    └── test_ranker.py
⚙️ Installation
1. Clone the Repository
git clone https://github.com/joyceangeleena21/SmartShop-AI.git

Move into the project directory:

cd SmartShop-AI
2. Install Dependencies
pip install -r requirements.txt
▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

Alternatively:

python -m streamlit run app.py

Then open the local URL displayed in the terminal, usually:

http://localhost:8501
🧪 Testing

The project includes automated tests for important components such as:

Dataset validation
Intent extraction
Embedding generation
Recommendation ranking
Explanation generation
End-to-end recommendation flow

Run the test suite using:

python -m unittest discover tests
📊 Example
User Query
I need a laptop under ₹70,000 for Python,
machine learning and college work.
Recommendation Output

The system returns ranked products and displays:

Product name
Price
Recommendation score
Semantic relevance
Budget compatibility
Rating
Specification match
Explanation for the recommendation

The user can then compare selected products using the Product Comparison Matrix.

🎯 Project Objectives

The main objectives of SmartShop AI are:

Understand natural-language shopping requirements
Improve product discovery beyond keyword-based search
Use semantic similarity for product matching
Combine multiple recommendation signals
Provide understandable recommendation explanations
Provide an interactive product comparison experience
💡 Key Learning Outcomes

This project demonstrates practical implementation of:

Natural-language requirement processing
Text embeddings
Semantic similarity
Recommendation systems
Hybrid ranking
Explainable AI
Data preprocessing
Streamlit application development
Automated testing
Modular Python project architecture
🔮 Future Improvements

Possible future enhancements include:

Larger and more diverse product datasets
Integration with real e-commerce product data
Conversational recommendation using an LLM
User preference and interaction history
Personalized recommendations
Product availability and price updates
Advanced recommendation evaluation
Cloud deployment
👩‍💻 Author

Joyce Angeleena Tera

B.Tech — CSE (Artificial Intelligence & Data Science)

GitHub:
https://github.com/joyceangeleena21
