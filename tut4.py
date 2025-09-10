# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# Load text using RELEVANT loader
from langchain_community.document_loaders import TextLoader
loader = TextLoader("book.txt", encoding="utf-8")
docs = loader.load()

# Split document into small chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split given book into {len(all_splits)} sub-documents.")

# Create embeddings with Google API key from .env
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Create a vector store
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

# Adding documents to vector store
document_ids = vector_store.add_documents(documents=all_splits)

# Extract chunks that match with your query
search_results = vector_store.similarity_search_with_score(
    "What does John Stuart Mill say about Auguste Comte’s idea of Positivism?",
    k=10
)

print(search_results)
