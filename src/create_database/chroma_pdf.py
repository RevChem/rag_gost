# python -m src.create_database.chroma_pdf
import os
from loguru import logger
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from ..config import settings
from src.create_database.pdf_processing import process_pdf
from src.create_database.semantic_chunking import merge_docs

def generate_chroma_db(pdf_dir: str, chunk_size: int, chunk_overlap: int):
    
    device = settings.DEVICE

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    texts, metadatas, ids = [], [], []

    if os.path.exists(settings.CHROMA_PATH):
        import shutil
        logger.info("Удаляется старая база данных...")
        shutil.rmtree(settings.CHROMA_PATH)

    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"Директория не найдена: {pdf_dir}")
    
    categories = [d for d in os.listdir(pdf_dir)]
    logger.info(f'Найдены категории: {categories}')

    for category in categories:
        folder_path = os.path.join(pdf_dir, category)
        logger.info(f'Начата обработка категории: {category}')

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith('.pdf'):
                continue

            file_path = os.path.join(folder_path, filename)
            logger.info(f"Обработка: {filename}")

            full_text = process_pdf(file_path)
            if not full_text.strip():
                continue

            raw_chunks = []
            for i in range(0, len(full_text), chunk_size - chunk_overlap):
                chunk = full_text[i:i + chunk_size]
                if len(chunk) < chunk_size and i > 0:
                    break
                raw_chunks.append(chunk.strip())

            raw_documents = [
                Document(page_content=chunk, metadata={"source": filename, "category": category})
                for chunk in raw_chunks if chunk
            ]

            merged_documents = merge_docs(raw_documents)

            for i, doc in enumerate(merged_documents):
                texts.append(doc.page_content)
                metadatas.append({
                    "source": filename,
                    "category": category,
                    "chunk": i
                })
                ids.append(f"{filename}_{i}")

    logger.info(f"Всего {len(texts)} фрагментов")


    chroma_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=settings.CHROMA_PATH,
        collection_name=settings.COLLECTION_NAME,
    )
    chroma_db.persist()
    logger.info("База создана.")
    return chroma_db


if __name__ == "__main__":
    generate_chroma_db('pdf/', 
                       chunk_size=settings.CHUNK_SIZE, 
                       chunk_overlap = settings.CHUNK_OVERLAP)