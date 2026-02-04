import json
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

INDEX_NAME = "quickcart_products"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_text(p: dict) -> str:
    return (
        f"Name: {p.get('name','')}\n"
        f"Brand: {p.get('brand','')}\n"
        f"Category: {p.get('category','')}\n"
        f"Price: {p.get('price','')}\n"
        f"Description: {p.get('description','')}\n"
    )

def main():
    # Load products
    with open("data/products.json", "r", encoding="utf-8") as f:
        products = json.load(f)

    if not products:
        raise RuntimeError("data/products.json is empty. Add at least 1 product.")

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    # Connect to Endee (defaults to http://localhost:8080)
    client = Endee()

    # Create index (if already exists, ignore)
    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=dim,
            space_type="cosine",
            precision=Precision.INT8D
        )
        print(f"Created index: {INDEX_NAME} (dim={dim})")
    except Exception as e:
        print(f"Index may already exist, continuing: {e}")

    index = client.get_index(name=INDEX_NAME)

    # Create vectors
    texts = [build_text(p) for p in products]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()

    # Upsert to Endee
    payload = []
    for p, v in zip(products, vectors):
        payload.append({
            "id": p["id"],
            "vector": v,
            "meta": {
                "name": p.get("name"),
                "brand": p.get("brand"),
                "price": p.get("price"),
                "category": p.get("category"),
                "description": p.get("description"),
            }
        })

    index.upsert(payload)
    print(f"Upserted {len(payload)} products into Endee index '{INDEX_NAME}'.")

if __name__ == "__main__":
    main()
