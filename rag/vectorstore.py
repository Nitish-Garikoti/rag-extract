import time

from pinecone import Pinecone as Pinecone_api, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from utils.config import settings

pc = Pinecone_api(api_key=settings.pinecone_api_key)


def create_pinecone_index(
    index_name: str,
    index_dimensions: int = settings.default_index_dimensions,
    metric: str = settings.default_index_metric,
    cloud: str = settings.default_index_cloud,
    region: str = settings.default_index_region,
):
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        dimension=index_dimensions,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)


def delete_pinecone_index(index_name: str):
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)


def create_vector_db(chunks, embeddings, index_name: str) -> PineconeVectorStore:
    vector_db = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
    return vector_db


def get_embeddings(embedding_model_name: str = settings.default_embedding_model) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=embedding_model_name)
