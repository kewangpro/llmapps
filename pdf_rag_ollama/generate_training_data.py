import json
import os
import random
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client

# Configuration
TEXT_SPLITTER_PARAMS = {
    "chunk_size": 400,
    "chunk_overlap": 100,
    "separators": ["\n\n", "\n", ".", "!", "?", " ", ""]
}
MODEL_NAME = "gemma3:latest"
OLLAMA_URL = "http://localhost:11434"
INPUT_FILES = ["README.md", "app.py", "vector.py", "llm.py", "ranker.py"]
OUTPUT_FILE = "training_data.json"

client = Client(host=OLLAMA_URL)

def generate_question(context: str) -> str:
    prompt = f"""
    Context: {context}
    
    Task: Generate a single, concise question that can be answered using the information in the context above. 
    The question should be natural and likely to be asked by a user of this PDF RAG application.
    Return ONLY the question text. Do not include any preamble or explanation.
    """
    response = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content.strip().strip('"')

def main():
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_PARAMS)

    print("Loading and splitting documents...")
    for filename in INPUT_FILES:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                content = f.read()
                splits = text_splitter.split_text(content)
                all_chunks.extend(splits)
    
    print(f"Total chunks created: {len(all_chunks)}")
    
    training_data = []
    
    # We'll generate a question for each chunk (positive pair)
    # And then pick random chunks as negative pairs
    for i, chunk in enumerate(all_chunks):
        print(f"Generating question for chunk {i+1}/{len(all_chunks)}...")
        try:
            query = generate_question(chunk)
            
            # Positive sample: (query, same chunk)
            texts = [chunk]
            labels = [1]
            
            # Negative samples: (query, different random chunks)
            # Pick 2 random chunks that are not the current one
            negative_indices = random.sample([idx for idx in range(len(all_chunks)) if idx != i], 2)
            for idx in negative_indices:
                texts.append(all_chunks[idx])
                labels.append(0)
                
            training_data.append({
                "query": query,
                "texts": texts,
                "labels": labels
            })
        except Exception as e:
            print(f"Error generating question for chunk {i}: {e}")

    print(f"Saving {len(training_data)} training examples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(training_data, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
