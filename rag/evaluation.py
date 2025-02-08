from operator import itemgetter

import pandas as pd
from datasets import Dataset
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity,
)

from rag.chain import answer_the_question, create_model_chain
from rag.vectorstore import create_pinecone_index, create_vector_db, delete_pinecone_index, get_embeddings
from utils.config import settings

GROUND_TRUTH_TEMPLATE = """
You are an expert in finance sector with great experience and knowledge in stock market.
With your expertise answer the Question below based on the Context provided
Context: {context}
Question: {question}
"""


def model_answers_dataset(chain, df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        answer = answer_the_question(chain, query=row["question"])
        records.append(
            {
                "question": row["question"],
                "answer": answer["response"],
                "contexts": [ctx.page_content for ctx in answer["context"]],
                "ground_truth": row["ground_truth"],
            }
        )
    return pd.DataFrame(records)


def evaluate_rag_chain_with_ragas(ragas_dataset: Dataset):
    return evaluate(
        ragas_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
            answer_correctness,
            answer_similarity,
        ],
    )


def rag_inference_and_eval(
    index_name: str,
    chunks,
    df_eval_dataset: pd.DataFrame,
    answer_synthesis_model_name: str = settings.default_model,
    answer_synthesis_model_temperature: float = settings.default_temperature,
    answer_synthesis_model_template: str = settings.answer_synthesis_template,
    index_dimensions: int = settings.default_index_dimensions,
    index_metric: str = settings.default_index_metric,
    index_cloud: str = settings.default_index_cloud,
    index_region: str = settings.default_index_region,
    embedding_model_name: str = settings.default_embedding_model,
    k: int = settings.default_k,
    search_type: str = settings.default_search_type,
):
    create_pinecone_index(index_name, index_dimensions, index_metric, index_cloud, index_region)
    embeddings = get_embeddings(embedding_model_name)
    vector_db = create_vector_db(chunks, embeddings, index_name)
    retriever = vector_db.as_retriever(search_type=search_type, search_kwargs={"k": k})
    chain = create_model_chain(retriever, answer_synthesis_model_name, answer_synthesis_model_temperature, answer_synthesis_model_template)
    df_rag = model_answers_dataset(chain, df_eval_dataset)
    rag_dataset = Dataset.from_pandas(df_rag)
    return evaluate_rag_chain_with_ragas(rag_dataset)


def get_ground_truth_dataset(df: pd.DataFrame, docs) -> pd.DataFrame:
    rcs = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=450, length_function=len, is_separator_regex=False)
    chunks = rcs.split_documents(docs)

    index_name = "pinecone-vdb-ground-truth"
    create_pinecone_index(index_name, 3072, "cosine", "aws", "us-east-1")
    embeddings = get_embeddings("text-embedding-3-large")
    vector_db = create_vector_db(chunks, embeddings, index_name)
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    gt_model = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate.from_template(GROUND_TRUTH_TEMPLATE)
    gt_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | gt_model | StrOutputParser(), "context": itemgetter("context")}
    )

    records = []
    for _, row in df.iterrows():
        result = gt_chain.invoke({"question": row["question"]})
        records.append({"question": row["question"], "ground_truth": result["response"]})

    delete_pinecone_index(index_name)
    return pd.DataFrame(records)
