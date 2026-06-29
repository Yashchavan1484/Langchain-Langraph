from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
import os

load_dotenv()

@tool
def triple(num:float):
    """
    param num: a number to triple
    returns: the triple of num
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"),temperature=0).bind_tools(tools)

