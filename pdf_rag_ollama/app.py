import os
import tempfile
import streamlit as st

from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from llm import call_llm
from vector import add_to_vector_collection, query_collection
from ranker import rerank_cross_encoders


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    # Store uploaded file as a temp file
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name) # Delete the temp file

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    return text_spliter.split_documents(docs)


if __name__ == "__main__":
    with st.sidebar:
        st.set_page_config(page_title="QnA")
        st.header("Sources")
        uploaded_file = st.file_uploader("** Upload PDF files for QnA **", type=["pdf"], accept_multiple_files=False)

        process = st.button("Process")

        if uploaded_file and process:
            normalized_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalized_uploaded_file_name)

    st.title("💬 PDF Understanding QnA")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Your question?"):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        results = query_collection(prompt)
        documents = results.get("documents")[0]
        context, relevant_doc_ids = rerank_cross_encoders(prompt, documents)

        stream_response = call_llm(context, prompt)

        response = st.chat_message("assistant").write_stream(stream_response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.expander("See retrieved document ids"):
            st.write(results.get("ids")[0])
            st.write("Most relevant:")
            st.write(relevant_doc_ids)
