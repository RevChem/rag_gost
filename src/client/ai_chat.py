from langchain_gigachat import GigaChat
from langchain_deepseek import ChatDeepSeek
from typing import AsyncGenerator, Literal
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import settings

class ChatWithAI:
    def __init__(self, provider: Literal['deepseek', 'gigachat'] = 'deepseek'):
        self.provider = provider
        if provider == 'deepseek':
            self.llm = ChatDeepSeek(
                api_key=settings.DEEPSEEK_API,
                model=settings.DEEPSEEK,
                temperature=0.0,
            )

        elif provider == 'gigachat':
            self.llm = GigaChat(
                model = 'Gigachat',
                verify_ssl_certs = False,
                temperature = 0.0
            )

        else:
            raise ValueError(f'Неподдерживаемый провайдер: {provider}')
        

    async def astream_response(
            self, formatted_context: str, query: str
    ) -> AsyncGenerator[str, None]:
        try:
            system_message = SystemMessage(
                content="""
        Ты — помощник химика. Отвечай кратко, основываясь лишь на переданном контексте. 
        Если требуется формула — оформляй её в блоке markdown с помощью `$$...$$` для отдельной строки или `$...$` для строки с текстом.

        Пример 1:
        Вопрос: Организация контроля питьевой воды.
        Ответ: Организация и проведение производственного контроля должны соответствовать требо­ваниям ГОСТ 2874, ГОСТ 2761 и национальным санитарно-эпидемиологическим правилам и нормам,
        установленным для предприятий пищевой промышленности, а также включать систему обеспечения и контроля качества.

        Пример 2 (с формулой):
        Вопрос: Как рассчитать молярную концентрацию раствора?
        Ответ: Молярная концентрация рассчитывается по формуле:
        $$ C = \frac{n}{V} $$
        где $C$ — молярная концентрация (моль/л), $n$ — количество вещества (моль), $V$ — объём раствора (л).
        """
        )
            human_message = HumanMessage(
                content = f'Вопрос: {query}\nКонтекст: {formatted_context}.'
            )

            async for chunk in self.llm.astream([system_message, human_message]):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            print(f'Error {e}')
            yield "Ошибка"
