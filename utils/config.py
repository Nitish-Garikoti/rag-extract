import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pdf_path: str = os.getenv("PDF_PATH", "./pdf_docs")
    eval_questions_path: str = os.getenv("EVAL_QUESTIONS_PATH", "./Evaluation_Questions.txt")
    outputs_dir: str = os.getenv("OUTPUTS_DIR", "./outputs")
    ground_truth_dir: str = os.getenv("GROUND_TRUTH_DIR", "./ground_truth")

    # Default RAG config
    default_model: str = "gpt-3.5-turbo"
    default_temperature: float = 0.5
    default_embedding_model: str = "text-embedding-3-large"
    default_index_dimensions: int = 3072
    default_index_metric: str = "cosine"
    default_index_cloud: str = "aws"
    default_index_region: str = "us-east-1"
    default_k: int = 10
    default_search_type: str = "similarity"

    answer_synthesis_template: str = """
You are an expert in finance sector with great experience and knowledge in stock market.
With your expertise answer the Question below based on the Context provided
Context: {context}
Question: {question}
"""

    class Config:
        env_file = ".env"


settings = Settings()
