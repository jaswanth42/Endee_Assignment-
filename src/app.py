import os
import re
from typing import Optional, List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from endee import Endee

from langchain_groq import ChatGroq

INDEX_NAME = "quickcart_products"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------- Helpers ----------
def connect_index():
    client = Endee()
    return client.get_index(name=INDEX_NAME)


def embed_query(model, q: str) -> List[float]:
    return model.encode([q], normalize_embeddings=True)[0].tolist()


def safe_price(meta: Dict[str, Any]) -> float:
    p = meta.get("price", 0)
    try:
        return float(p)
    except Exception:
        return 0.0


def extract_max_price(query: str) -> Optional[float]:
    q = query.lower().strip()

    # "under 10k", "below 15k"
    k_match = re.search(r"(below|under|less than|<=|<)\s*(\d+)\s*k", q)
    if k_match:
        return float(k_match.group(2)) * 1000

    # "under 10000", "below 15000"
    num_match = re.search(r"(below|under|less than|<=|<)\s*(\d{3,7})", q)
    if num_match:
        return float(num_match.group(2))

    return None


def apply_filters(results: List[dict], max_price: Optional[float]) -> List[dict]:
    if max_price is None:
        return results
    return [r for r in results if safe_price(r.get("meta", {}) or {}) <= max_price]


def apply_sort(results: List[dict], sort_mode: str) -> List[dict]:
    if sort_mode == "Price: Low to High":
        return sorted(results, key=lambda r: safe_price((r.get("meta", {}) or {})))
    if sort_mode == "Price: High to Low":
        return sorted(results, key=lambda r: safe_price((r.get("meta", {}) or {})), reverse=True)
    return results  # semantic relevance


def render_product_cards(results: List[dict]):
    for r in results[:10]:
        meta = r.get("meta", {}) or {}
        name = meta.get("name", "N/A")
        price = meta.get("price", "N/A")
        category = meta.get("category", "N/A")
        brand = meta.get("brand", "")
        desc = meta.get("description", "")

        with st.container(border=True):
            st.markdown(f"### {name}")
            st.markdown(f"**₹{price}**  |  **{category}**" + (f"  |  **{brand}**" if brand else ""))
            if desc:
                st.caption(desc)


def fallback_answer(user_query: str, results: List[dict], max_price: Optional[float]) -> str:
    if not results:
        if max_price is not None:
            return f"I couldn't find products under ₹{int(max_price)}. Try increasing budget or changing query."
        return "I couldn't find relevant products. Try a different query."

    lines = []
    lines.append(f"Here’s what I found for: **{user_query}**")
    if max_price is not None:
        lines.append(f"Applied filter: **price <= ₹{int(max_price)}**")
    lines.append("")
    lines.append("**Top recommendations:**")
    for i, r in enumerate(results[:3], start=1):
        meta = r.get("meta", {}) or {}
        lines.append(f"{i}. **{meta.get('name','N/A')}** (₹{meta.get('price','N/A')}, {meta.get('category','N/A')})")
    return "\n".join(lines)


def groq_rag_answer(user_query: str, results: List[dict], max_price: Optional[float]) -> str:
    """
    RAG-style answer:
    - Retrieval from Endee already done
    - We only send retrieved products as context to Groq
    - Groq generates a grounded recommendation response
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        return fallback_answer(user_query, results, max_price)

    if not results:
        return fallback_answer(user_query, results, max_price)

    # Prepare context from retrieved products
    context_lines = []
    for r in results[:8]:
        meta = r.get("meta", {}) or {}
        context_lines.append(
            f"- name: {meta.get('name','')}, price: {meta.get('price','')}, category: {meta.get('category','')}, description: {meta.get('description','')}"
        )
    context = "\n".join(context_lines)

    budget_text = f"User budget: under ₹{int(max_price)}" if max_price is not None else "User budget: not specified"

    prompt = f"""
You are an e-commerce shopping assistant for QuickCart.

Important:
- You must ONLY use the products listed in the CONTEXT (retrieved by vector search).
- If a budget is given, do NOT recommend items above the budget.
- Return a short, helpful answer with 2-3 recommendations and reasons.

User query: {user_query}
{budget_text}

CONTEXT (retrieved products):
{context}

Output format:
- Recommended items (2-3 bullets)
- Why (short reason per item)
- If no product matches the budget, clearly say so.
"""

    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        groq_api_key=groq_key
    )

    return llm.invoke(prompt).content


# ---------- Streamlit App ----------
load_dotenv()

st.set_page_config(page_title="QuickCart Chat", page_icon="🛒", layout="wide")
st.title("🛒 QuickCart Chat")
st.caption("Chatbot UI. Semantic retrieval powered by Endee. AI response powered by Groq (optional).")

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()

# Connect to Endee
try:
    index = connect_index()
    st.success(f"Connected to Endee index: {INDEX_NAME}")
except Exception as e:
    st.error("Could not connect to Endee. Make sure Endee is running and you ran ingest.py.")
    st.code(str(e))
    st.stop()

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Controls")
    top_k = st.slider("Top K Retrieval", 3, 20, 10, 1)
    sort_mode = st.selectbox("Sort Results", ["Semantic relevance", "Price: Low to High", "Price: High to Low"])
    use_ai = st.checkbox("Use Groq AI response (RAG-style)", value=True)
    st.caption("Tip: queries like `phones under 15000`, `list products below 10k`")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me to find products. Example: `phones under 15000 with good camera`."}
    ]

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_query = st.chat_input("Type your query…")
if user_query and user_query.strip():
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        max_price = extract_max_price(user_query)

        q_vec = embed_query(model, user_query)
        results = index.query(vector=q_vec, top_k=top_k)

        results = apply_filters(results, max_price)
        results = apply_sort(results, sort_mode)

        # AI answer
        if use_ai:
            reply = groq_rag_answer(user_query, results, max_price)
        else:
            reply = fallback_answer(user_query, results, max_price)

        st.markdown(reply)

        st.markdown("#### Results")
        if results:
            render_product_cards(results)
        else:
            st.info("No results after applying filters.")

    st.session_state.messages.append({"role": "assistant", "content": reply})
