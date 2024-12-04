from sentence_transformers import CrossEncoder

def rerank_cross_encoders(prompt: str, documents: list[str]) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)

    for rank in ranks:
        id = rank["corpus_id"]
        relevant_text += documents[id]
        relevant_text_ids.append(id)

    return relevant_text, relevant_text_ids
