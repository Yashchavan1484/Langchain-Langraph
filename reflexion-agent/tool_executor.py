from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer


tavily_tool = TavilySearch(max_results=3)


def run_queries(search_queries: list[str], **kwargs):
    """
    Search the web using Tavily for the provided queries.
    Returns relevant research information.
    """

    results = tavily_tool.batch(
        [{"query": q} for q in search_queries]
    )

    return results


execute_tools = ToolNode(
    [
        StructuredTool.from_function(
            run_queries,
            name=AnswerQuestion.__name__
        ),

        StructuredTool.from_function(
            run_queries,
            name=ReviseAnswer.__name__
        ),
    ]
)