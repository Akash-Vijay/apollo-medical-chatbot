from langchain_chroma import Chroma
from src.helper import load_pdf, text_split, download_hugging_face_embeddings

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embedding_model = download_hugging_face_embeddings()

db = Chroma.from_documents(text_chunks, embedding_model, persist_directory="./chroma_db")

