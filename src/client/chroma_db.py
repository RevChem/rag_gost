import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.config import settings

class ChromaDatabase:
    def __init__(self):
        self.store: Chroma | None = None

    async def init(self):
        """Инициализация бд Chroma."""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            embeddings = HuggingFaceEmbeddings(
                model_name=settings.LM_MODEL_NAME,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )

            self.store = Chroma(
                persist_directory=settings.CHROMA_PATH,
                embedding_function=embeddings,
                collection_name=settings.COLLECTION_NAME,
                collection_metadata={'hnsw:space': 'cosine'},
            )

            logger.success(f'Подключение к коллекции {settings.COLLECTION_NAME}')
        except Exception as e:
            logger.exception(f'Ошибка при инициализации Chroma: {e}')
            raise


    async def search_document(self, query: str, filter: dict | None = None, with_score=True, k: int = 3):
        """Поиск документа в Chroma."""
        if not self.store:
            raise RuntimeError('Хранилище не инициализировано')
        
        logger.info(f'Поиск документов по запросу: {query}')

        try:
            if with_score:
                results = self.store.similarity_search_with_score(query, k=k)
            else:
                results = self.store.similarity_search(query, k=k)

            logger.debug(f'Найдено {len(results)} документов')
            return results
        except Exception as e:
            logger.exception(f'Ошибка поиска: {e}')
            raise


    def close(self):
        logger.info("Отключение Chroma")
        pass


chroma_database = ChromaDatabase()


def get_chroma_database() -> ChromaDatabase:
    return chroma_database
