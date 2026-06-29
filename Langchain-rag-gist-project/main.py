import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore 

load_dotenv()
print("Initializing components...")

embeddings = GoogleGenerativeAIEmbeddings( model="models/gemini-embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"])
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=os.environ["GEMINI_API_KEY"])

vector_store = PineconeVectorStore(embedding=embeddings,index_name=os.environ["INDEX_NAME"])

retriever = vector_store.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}

    Question : {question}

    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrieved documents into a single string"""

    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query:str):
    """
    Simple retrieval chain without LECL.
    Manually retrives documents ,formats them and generates response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    docs = retriever.invoke(query)

    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context,question = query)

    response = llm.invoke(messages)

    return response.content

def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain



if __name__ == "__main__":
    print("Retrieving...")

    query = "what is Pinecone in machine learning?"

    print("\n"+"\n"+"="*70)
    # result_raw = llm.invoke(query)
    # result_without_lecl = retrieval_chain_without_lcel(query)
    chain_with_lecl = create_retrieval_chain_with_lcel()
    result_with_lecl = chain_with_lecl.invoke({"question":query})
    print(result_with_lecl)