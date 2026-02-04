from sentence_transformers import SentenceTransformer
from endee import Endee

INDEX_NAME = "quickcart_products"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def is_price_sort_query(q: str) -> bool:
    q = q.lower()
    keywords = [
        "price", "cost", "cheapest", "expensive",
        "low to high", "high to low", "sort"
    ]
    return any(k in q for k in keywords)


def wants_desc_sort(q: str) -> bool:
    q = q.lower()
    return ("high to low" in q) or ("expensive" in q) or ("desc" in q) or ("descending" in q)


def main():
    model = SentenceTransformer(MODEL_NAME)
    client = Endee()
    index = client.get_index(name=INDEX_NAME)

    print("QuickCart Semantic Search (Endee)")
    print("Type a query and press Enter. Type 'exit' to quit.\n")

    while True:
        q = input("Search (type exit): ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break

        # Embed the query
        q_vec = model.encode([q], normalize_embeddings=True)[0].tolist()

        # Retrieve from Endee (top_k can be increased if you add more products)
        results = index.query(vector=q_vec, top_k=10)

        if not results:
            print("\nNo matches found.")
            continue

        # Sort by price if the query asks for it
        if is_price_sort_query(q):
            desc = wants_desc_sort(q)

            def price_key(r):
                meta = r.get("meta", {}) or {}
                price = meta.get("price", 0)
                try:
                    return float(price)
                except Exception:
                    return 0.0

            results = sorted(results, key=price_key, reverse=desc)

            order_label = "High → Low" if desc else "Low → High"
            print(f"\nProducts sorted by price ({order_label}):")
        else:
            print("\nTop semantic matches:")

        # Print results
        for r in results:
            meta = r.get("meta", {}) or {}
            name = meta.get("name", "N/A")
            price = meta.get("price", "N/A")
            category = meta.get("category", "N/A")
            brand = meta.get("brand", "")
            brand_part = f" | {brand}" if brand else ""
            print(f"- {name}{brand_part} | ₹{price} | {category}")


if __name__ == "__main__":
    main()
