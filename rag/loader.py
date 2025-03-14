from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_documents(pdf_path: str):
    document_loader = PyPDFDirectoryLoader(pdf_path)
    return document_loader.load()
