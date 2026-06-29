import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore 

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    # print(os.environ["PINECONE_API_KEY"])
    loader = TextLoader(
    r"E:\langchain-course\mediumblog1.txt",
    encoding="utf-8",
    autodetect_encoding=True)
    document = loader.load()
    print("splitting")
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    text = text_splitter.split_documents(document)
    print(f"created {len(text)} chunks.")

    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"]
    )

    print("ingesting")

    PineconeVectorStore.from_documents(text,embeddings,index_name=os.environ["INDEX_NAME"])
    print("finish!")