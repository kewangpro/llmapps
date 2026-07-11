import os
from sentence_transformers import CrossEncoder

def rerank_cross_encoders(prompt: str, documents: list[str]) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []

    # Check for locally trained model
    model_path = "fine-tuned-ranker"
    if os.path.exists(model_path):
        print(f"Loading custom model from {model_path}")
        encoder_model = CrossEncoder(model_path)
    else:
        print("Loading default cross-encoder model")
        encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
    ranks = encoder_model.rank(prompt, documents, top_k=3)

    for rank in ranks:
        id = rank["corpus_id"]
        relevant_text += documents[id]
        relevant_text_ids.append(id)

    return relevant_text, relevant_text_ids
