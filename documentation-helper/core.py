import os
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


# ----------------------------
# Embeddings
# ----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GEMINI_API_KEY"]
)

# ----------------------------
# Vector Store
# ----------------------------
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index",
    embedding=embeddings
)

# ----------------------------
# Retriever
# ----------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# ----------------------------
# LLM
# ----------------------------
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai"
)

# ----------------------------
# Tool
# ----------------------------
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve relevant documentation to answer LangChain questions.
    """

    retrieved_docs = retriever.invoke(query)

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\nDocument {i}")
        print("Source:", doc.metadata.get("source", "Unknown"))
        print(doc.page_content[:300])
        print("-" * 40)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent:\n{doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


# ----------------------------
# RAG Pipeline
# ----------------------------
def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the Agentic RAG pipeline.
    """

    system_prompt = """
    You are a helpful AI assistant that answers questions about LangChain.

    You have access to a retrieval tool that fetches relevant documentation.

    Instructions:
    1. Always use the retrieval tool before answering.
    2. Base your answer only on the retrieved documentation.
    3. Cite the source URLs whenever possible.
    4. If the answer is not present in the retrieved documents,
       clearly state that you could not find it.
    """

    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=system_prompt
    )

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
    )

    # ----------------------------
    # Extract Answer
    # ----------------------------
    last_message = response["messages"][-1]

    if isinstance(last_message.content, list):
        answer_parts = []

        for block in last_message.content:
            if isinstance(block, dict):
                answer_parts.append(block.get("text", ""))

        answer = "\n".join(answer_parts)

    else:
        answer = str(last_message.content)

    # ----------------------------
    # Extract Retrieved Docs
    # ----------------------------
    context_docs: List[Document] = []

    for message in response["messages"]:
        if isinstance(message, ToolMessage):
            if hasattr(message, "artifact"):
                if isinstance(message.artifact, list):
                    context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":

    result = run_llm(
        query="What are Deep Agents?"
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(result["answer"])

    print("\n" + "=" * 60)
    print(f"Retrieved {len(result['context'])} documents")
    print("=" * 60)