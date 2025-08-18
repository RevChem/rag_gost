import torch
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek

from src.config import settings
from src.client.ai_chat import ChatWithAI
from src.client.chroma_db import get_chroma_database


critic_llm = LangchainLLMWrapper(ChatDeepSeek(
    api_key=settings.DEEPSEEK_API,
    model=settings.DEEPSEEK,
))

embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
))

questions = [
    {
        "question": "Как приготовить раствор гидроокиси натрия концентрации 330 г/дм**3?",
        "category": "мясные продукты"
    },
    {
        "question": "Массовая доля углеводов мяса категории А",
        "category": "мясные продукты"
    },
    {
        "question": "Определение истинного белка",
        "category": "молочные продукты"
    },
    {
        "question": "Как рассчитывается массовая доля белка в сухом обезжиренном молочном остатке для сухого молока?",
        "category": "молочные продукты"
    },
    {
        "question": "Что используется для хранения проб, содержащих светочувствительные материалы?",
        "category": "вода питьевая"
    },
    {
        "question": "При каких условиях хранят емкости с водой в транспортной упаковке?",
        "category": "вода питьевая"
    },
    {
        "question": "Какова объемная доля этанола в виноградной водке?",
        "category": "водка"
    },
    {
        "question": "Как приготовить яично-желточно-азидный агар",
        "category": "мясо птицы"
    },
]

ground_truths = [
    "Растворяют 330 г гидроокиси натрия в 200— 300 см3 дистиллированной воды, количественно пе­реносят в мерную колбу вместимостью 1000 см3 и доводят объем дистиллированной водой до метки. Раствор хранят при температуре (20 ± 2) °С не более 1 мес.",
    "Не более 3,0%",
    "Разность между массовой долей общего азота и небелкового азота, умно­женная на коэффициент 6,38.",
    "$$X_б = (X_об / X_сомо) - 100$$",
    "Для хранения проб, содержащих светочувствительные ингредиенты (включая морские водо­росли), применяют емкости из светонепроницаемого или неактиничного стекла с последующим разме­щением их в светонепроницаемую упаковку на весь период хранения проб.",
    "Емкости с водой, упакованные в транспортную тару (например, по ГОСТ 23285), хранят в про­ветриваемых затемненных складских помещениях при температуре от 2 °С до 20 °С и относительной влажности не выше 85 %",
    "Объемная доля этилового спирта в виноградной водке с учетом допустимых отклонений должна быть не менее 37.5 %",
    "В 1000 см3 дистиллированной воды растворяют при нагревании 10 г пептона, 3 г хлористого на­трия, 0.2 г фосфорнокислого деузамещенного натрия (дигидрофосфат). 15 г микробиологического ага­ра. 5.5 г мясного экстракта. При отсутствии мясного экстракта используют мясную воду (11.4.36). приэтом все компоненты растворяют в 1000 см3 мясной воды. Устанавливают pH (7,610.1) ед. pH. Стери­лизуют в течение 30 мин при температуре (121 ± 1)С. охлаждают до температуры (50—60) *С. добав­ляют 0,15 г азида натрия, смешивают, вновь стерилизуют в течение 30 мин при температуре (121 ♦ 1) *С. После охлаждения до температуры (5011) вС добавляют 150 см3 желточной эмульсии (11.4.23). смешивают и разливают в чашки Петри. Срок хранения — не более 5 сут при температуре (411) *С"
]

async def run_evaluation():
    vectorstore = get_chroma_database()
    if vectorstore.store is None:
        await vectorstore.init()

    results = []
    for i, item in enumerate(questions):
        query = item["question"] 
        result = await vectorstore.search_document(query=query, with_score=True, k=5)
        contexts = [doc.page_content for doc, _ in result]
        ai_context = "\n".join(contexts)

        ai_store = ChatWithAI() 
        response_chunks = []
        async for chunk in ai_store.astream_response(ai_context, query):  
            response_chunks.append(chunk)
        answer = ''.join(response_chunks)
        print(f"Вопрос: {query}\nОтвет: {answer}\n")

        results.append({
            "question": query,  
            "answer": answer,
            "contexts": contexts,
            "reference": ground_truths[i] 
        })

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "reference": [r["reference"] for r in results],
    })

    metrics = [
        faithfulness,
        answer_relevancy,
        answer_correctness,
        context_precision,
    ]

    scores = evaluate(
        dataset,
        metrics=metrics,
        llm=critic_llm,
        embeddings=embeddings,
    )
    print(scores)
    return scores


if __name__ == "__main__":
    asyncio.run(run_evaluation())