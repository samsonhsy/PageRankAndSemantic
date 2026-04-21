from sentence_transformers import SentenceTransformer
from pathlib import Path
from bs4 import BeautifulSoup
import json
import numpy as np

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
def get_embeddings(model_name="all-MiniLM-L6-v2"):
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model


def extract_html_text(html_path):
    html = Path(html_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    parts = []
    if soup.title and soup.title.get_text(strip=True):
        parts.append(soup.title.get_text(" ", strip=True))
    for tag in soup.select("h1, h2, p"):
        txt = tag.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    text = " ".join(parts)
    text = " ".join(text.split())
    return text

def get_html_embedding(cache_path=Path("./data/html_embedding.json")):
    # load cache first
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                embedding_dict = json.load(f)
            if embedding_dict:
                print("Loaded cached embeddings from html_embedding.json")
                # print(embedding_dict)
                return embedding_dict
        except json.JSONDecodeError:
            pass  

    # compute embeddings when cache empty or invalid
    embedding_dict = {}
    model = get_embeddings() 
    for html_file in sorted(Path("./data").glob("*.html")):
        page_text = extract_html_text(html_file)
        # print("\npage_text:\n", page_text)
        embeddings = model.encode(page_text)  
        embedding_dict[html_file.name] = embeddings.tolist()

    # save cache
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(embedding_dict, f, ensure_ascii=True, indent=2)
    print("Saved embeddings to ./data/html_embedding.json")
    # print(embedding_dict)
    return embedding_dict

def cosine_similarity(vec_a, vec_b):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    dot_product = np.dot(vec_a, vec_b)
    
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    return dot_product / (norm_a * norm_b)

def rank_by_query(query, embedding_dict):
    model = get_embeddings() 
    query_embeddings = model.encode(query)  
    print("query_embeddings", query_embeddings)
    similarity = {}
    for page, vector in embedding_dict.items():
        similarity[page] = cosine_similarity(vector, query_embeddings)
    pages_sorted_desc = dict(sorted(similarity.items(), key=lambda item: item[1], reverse=True))
    print(pages_sorted_desc)
    return pages_sorted_desc


if __name__ == "__main__":
    query = "machine learning basics"
    rank_by_query(query, get_html_embedding())