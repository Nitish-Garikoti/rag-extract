from langchain.document_loaders.pdf import PyPDFDirectoryLoader


def load_documents(pdf_path: str):
    document_loader = PyPDFDirectoryLoader(pdf_path)
    return document_loader.load()
