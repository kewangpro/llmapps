# 🧠 PDF RAG with Ollama & Intelligent Re-ranking

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Powered%20by-Ollama-orange.svg)](https://ollama.ai/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A state-of-the-art **Retrieval-Augmented Generation (RAG)** application designed for private, local-first document intelligence. Chat with your PDFs and web content using powerful local LLMs, backed by ChromaDB vector storage and a fine-tunable Cross-Encoder re-ranker.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Local-First** | Complete privacy. No data leaves your machine, powered by Ollama. |
| **Hybrid Processing** | Seamlessly ingest PDF documents and Web URLs. |
| **Advanced Retrieval** | Two-stage retrieval: Vector search (ChromaDB) + Cross-Encoder re-ranking. |
| **Fine-tuneable Ranker** | Custom training script to optimize retrieval for your specific domain. |
| **Interactive UI** | Beautiful, responsive Streamlit interface with real-time response streaming. |

---

## 🏗️ Architecture

```text
       ┌───────────────┐          ┌────────────────┐
       │ PDF Documents │          │   Web URLs     │
       └───────┬───────┘          └────────┬───────┘
               │                           │
               ▼                           ▼
       ┌───────────────────────────────────────────┐
       │   Text Splitting & Embedding (Ollama)     │
       └─────────────────────┬─────────────────────┘
                             │
                             ▼
       ┌───────────────────────────────────────────┐
       │        Vector Search (ChromaDB)           │
       └─────────────────────┬─────────────────────┘
                             │
                             ▼
       ┌───────────────────────────────────────────┐
       │      Intelligent Re-ranking (BERT)        │
       │    (Custom or Default Cross-Encoder)      │
       └─────────────────────┬─────────────────────┘
                             │
                             ▼
       ┌───────────────────────────────────────────┐
       │        LLM Generation (Ollama)            │
       │           (gemma3:latest)                 │
       └─────────────────────┬─────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Chat Response   │
                    └─────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites

*   **Python**: 3.9 or higher.
*   **Ollama**: Install from [ollama.ai](https://ollama.ai/).
*   **System Libs (macOS)**:
    ```bash
    brew install libheif poppler tesseract-lang
    ```

### 2. Model Setup
```bash
ollama pull gemma3:latest
ollama pull nomic-embed-text:latest
```

### 3. Installation
```bash
# Clone and enter directory
git clone <repository-url>
cd pdf_rag_ollama

# Setup Virtual Environment
python -m venv .venv
source .venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 🎯 Re-ranker Fine-tuning

Improve retrieval accuracy by training the re-ranker on your own data.

1.  **Prepare Training Data**: Edit `training_data.json` with domain-specific queries and document pairs.
    > [!TIP]
    > If you don't have enough data, you can use the `generate_training_data.py` script to create synthetic training pairs from your existing project files and documentation.
2.  **Train the Model**:
    ```bash
    python3 train_ranker.py
    ```
3.  **Deploy**: The application automatically detects the `fine-tuned-ranker/` directory and prioritizes it over the default model.

---

## 🛠️ Configuration & Customization

The system is designed to be highly modular and robust:

- **Database Persistence**: ChromaDB uses absolute paths (managed in `vector.py`) to ensure reliable disk access across different environments and virtual environments.
- **Python Compatibility**: The project is optimized for Python 3.9+, with specific version pins like `pi-heif<0.15.0` to maintain backward compatibility.

The system is designed to be highly modular. You can easily swap components:

| Component | File | Variable | Default |
| :--- | :--- | :--- | :--- |
| **LLM Model** | `llm.py` | `model` | `gemma3:latest` |
| **Embeddings** | `vector.py` | `model_name` | `nomic-embed-text:latest` |
| **Chunk Size** | `app.py` | `chunk_size` | `400` |
| **Re-ranker** | `ranker.py` | `model_path` | `ms-marco-MiniLM-L-6-v2` |

---

## 📂 Project Structure

*   `app.py`: Main Streamlit application and orchestration.
*   `vector.py`: ChromaDB integration and embedding logic.
*   `llm.py`: Ollama API client for response generation.
*   `ranker.py`: Document re-ranking using Cross-Encoders.
*   `train_ranker.py`: Utility for fine-tuning the re-ranker model.
*   `training_data.json`: Template for fine-tuning datasets.

---

## 🔧 Troubleshooting

*   **Ollama Connection**: Ensure `ollama serve` is running.
*   **Missing Dependencies**: Ensure you've run the `brew install` command above for PDF processing.
*   **Memory Issues**: For smaller RAM, try lowering the `chunk_size` in `app.py` or using a smaller model like `phi4`.

---

## 📜 License

This project is open-source. See the license file for more details.