from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding=OllamaEmbeddings(model="nomic-embed-text")
loader=DirectoryLoader(
    path=r'D:\Skyclad_Ventures_assignment\arxiv_corpus', 
    glob="**/*.pdf",           
    loader_cls=PyPDFLoader,     
    show_progress=True
)
documents=loader.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=150)
chunks=splitter.split_documents(documents)
vector=FAISS.from_documents(chunks, embedding)
vector.save_local("faiss_index")