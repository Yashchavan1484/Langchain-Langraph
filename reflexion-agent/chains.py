import datetime

from dotenv import load_dotenv
import os

load_dotenv()


from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from schemas import AnswerQuestion,ReviseAnswer

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"),temperature=0)

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(
    tools=[
        AnswerQuestion,
        ReviseAnswer
    ]
)



actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

{first_instruction}

Return ONLY the required tool schema.
Do not add extra text.
Do not write Reflection: or Search Queries: labels.
The tool fields already define the structure.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="""
    Answer the user question in about 250 words.

    You must provide:
    - answer
    - reflection with missing and superfluous fields
    - 1-3 search_queries

    Do not add explanations outside the tool call.
    """
)

first_responder = (
    first_responder_prompt_template
    | llm.bind_tools(
        [AnswerQuestion],
        tool_choice={
            "type":"function",
            "function":{
                "name":"AnswerQuestion"
            }
        }
    )
)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
                " list startups that do that and raised capital."
    )
    chain = (
            first_responder_prompt_template
            | llm.bind_tools([AnswerQuestion])
            | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)

