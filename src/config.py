import os
import torch
from typing import ClassVar
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):

    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    model_config = SettingsConfigDict(env_file=f"{BASE_DIR}/../.env")

    CHROMA_PATH: str = os.path.join(BASE_DIR, "gost_database")
    COLLECTION_NAME: str = "docs"
    pdf_dir: str = 'pdf/'
    DEVICE: ClassVar[str] = "cuda" if torch.cuda.is_available() else "cpu"

    LM_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    GIGACHAT: str = "GigaChat" 
    GIGACHAT_CREDENTIALS: SecretStr

    DEEPSEEK: str = 'deepseek-chat'
    DEEPSEEK_API: SecretStr

    HF_MODEL_NAME: str = "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"  
    HF_API_TOKEN: SecretStr
    
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
settings = Config()  
