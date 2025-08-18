# Поправка!
# Здесь я был вынужден скачивать и использовать локальные версии токенайзера и модели, т.к.
# по невыясненной причине они не загружались из hf. Вы можете заменить их на закомментированные версии
from langchain_core.documents import Document
from transformers import AutoTokenizer, pipeline
import numpy as np
from tqdm import tqdm

# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/tokenizer/")


# nli_model = pipeline(
#     "zero-shot-classification",
#     model="cointegrated/rubert-tiny2"
# )
nli_model = pipeline(
    "zero-shot-classification",
    model="model/nli_model/nli_model/"
)

def count_tokens(text: str) -> int:
    """Подсчёт количества токенов в тексте."""
    return len(tokenizer.encode(text))

def same_sentence(premise: Document, hypothesis: Document, zero_shot_classifier) -> bool:
    """Проверка, относятся ли premise и hypothesis к одному предложению."""
    input_text = f"{premise.page_content}\n\n{hypothesis.page_content}"
    candidate_labels = ['в одном предложении', 'в разных предложениях']
    output = zero_shot_classifier(input_text, candidate_labels, multi_label=False)
    true_idx = output['labels'].index('в одном предложении')
    pred = np.argmax(output['scores']) == true_idx
    return pred

def merge_docs(chunks: list[Document], nli_model = nli_model) -> list[Document]:
    """Объединение чанков, которые относятся к одному предложению."""
    if not chunks:
        return []

    merged_chunks = []
    premise = chunks[0]

    def _add_chunk(chunk: Document):
        merged_chunks.append(chunk)

    for i in tqdm(range(1, len(chunks))):
        hypothesis = chunks[i]
        if same_sentence(premise, hypothesis, nli_model):
            premise_candidate = premise.page_content + ' ' + hypothesis.page_content
            if count_tokens(premise_candidate) > 500:
                _add_chunk(premise)
                premise = hypothesis
            else:
                premise = Document(page_content=premise_candidate, metadata=premise.metadata)
        else:
            _add_chunk(premise)
            premise = hypothesis

    _add_chunk(premise)
    return merged_chunks