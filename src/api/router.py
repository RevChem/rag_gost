from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.schemas import AskWithAIResponse
from src.model.ai import AI_Settings
from src.chroma_database.search_documents import ChromaDatabase, get_chroma_database


router = APIRouter()

@router.post('/ask')
async def ask(
    query: str,
    category: str | None = None,
    vectorstore: ChromaDatabase = Depends(get_chroma_database),
):
    
    filter = {'category': category} if category else None
    results = await vectorstore.search_document(
        query  = query, filter = filter, with_score = True, k=5, 
    )

    formatted_results = []

    for doc, score in results:

        formatted_results.append(
            {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': score,
            }
        )
    return {'results': formatted_results}


@router.post("/ask_with_ai")
async def ask_with_ai(
    query: str,
    category: str | None,
    vectorstore: ChromaDatabase = Depends(get_chroma_database),
):
    
    filter = {'category': category} if category else None
    results = await vectorstore.search_document(
    query  = query, filter = filter, with_score = True, k=5, 
    )
    if results:
        ai_context = "\n".join([doc.page_content for doc, _ in results])
        ai_store = AI_Settings()  

        async def stream_response():
            async for chunk in ai_store.astream_response(ai_context, query):
                yield chunk

        return StreamingResponse(
            stream_response(),
            media_type="text/plain",
            headers={
                "Content-Type": "text/plain",
                "Transfer-Encoding": "chunked",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        return {"response": "Ничего не найдено"}