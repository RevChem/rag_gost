# python -m src.processing.chroma_pdf
import os
from loguru import logger
from ..config import settings
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


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

            elements = partition_pdf(
                filename=file_path,
                # Можно заменить fast на 'hi_res', если pdf изобилует таблицами или имеет сложную структуру
                # или 'ocr_only', если документ сканирован
                strategy='fast',
                infer_table_structure=True,
                include_page_breaks=False,
                languages=["rus"]
            )

            chunks = chunk_elements(
                elements, 
                max_characters=chunk_size, 
                overlap=chunk_overlap
            )

            raw_documents = [Document(page_content=chunk.text) for chunk in chunks]

            for i, doc in enumerate(raw_documents):
                text = doc.page_content.strip()
                texts.append(text)
                metadatas.append({
                    "source": filename,
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