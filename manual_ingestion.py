import os
import glob
from multiprocessing import Pool
from tqdm import tqdm
from typing import List
from langchain.document_loaders import (DirectoryLoader, PyPDFLoader, CSVLoader, TextLoader, UnstructuredHTMLLoader,
                                        UnstructuredMarkdownLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.docstore.document import Document
import pinecone
from dotenv import load_dotenv
import os

# load your credentials from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index = os.getenv("PINECONE_INDEX_NAME")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0

"""
Ingest your documents into Pinecone vectorstore
"""

source_directory = 'docs'  # path to folder containing documents to ingest

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    """
    load a single document from a file path
    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    print(
        f"Split into {len(texts)} chunks of text (max. {CHUNK_SIZE} tokens each)")
    return texts


def ingest_docs():
    # throw error if environment variables are not set
    env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY",
                "PINECONE_ENVIRONMENT", "PINECONE_INDEX_NAME", "PINECONE_NAMESPACE"]

    for var in env_vars:
        if not os.getenv(var):
            raise ValueError(f"Please set {var} in .env file.")

    # initialize pinecone if not already initialized
    pinecone.init(
        api_key=pinecone_api_key, environment=pinecone_environment)

    try:
        print("Ingesting documents into Pinecone vectorstore...")
        # load documents from folder and split in chunks
        texts = process_documents()

        # # create embeddings
        embeddings = OpenAIEmbeddings(
            model='text-embedding-ada-002', openai_api_key=openai_api_key)

        # # ingest documents into pinecone
        Pinecone.from_documents(
            texts, embeddings, index_name=pinecone_index, namespace=pinecone_namespace)
        print("Documents ingested into Pinecone vectorstore.")
        return True
    except Exception as e:
        print(f"An error occurred whilst ingesting your files: {str(e)}")
        raise e


if __name__ == "__main__":
    ingest_docs()
