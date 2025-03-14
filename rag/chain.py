from operator import itemgetter

from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils.config import settings


def create_model_chain(
    retriever,
    model_name: str = settings.default_model,
    temperature: float = settings.default_temperature,
    template: str = settings.answer_synthesis_template,
):
    chat_model = ChatOpenAI(model_name=model_name, temperature=temperature)
    prompt = PromptTemplate.from_template(template)
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {
            "response": prompt | chat_model | StrOutputParser(),
            "context": itemgetter("context"),
        }
    )
    return chain


def answer_the_question(chain, query: str) -> dict:
    return chain.invoke({"question": query})
