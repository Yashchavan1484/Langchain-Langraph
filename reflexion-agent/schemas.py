from typing import List

from pydantic import BaseModel,Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")

class AnswerQuestion(BaseModel):
    """Answer the question"""

    answer: str = Field(description="~250 words detailed answer to the question.")
    reflection: Reflection = Field(description="Reflection on the answer.")

    search_queries: List[str] = Field(description="1-3 search queries for researching improvements to address the critique of your answer.")

class ReviseAnswer(BaseModel):
    """Revise your original answer to your question."""

    answer: str = Field(
        description="Improved final answer around 250 words."
    )

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )

    search_queries: List[str] = Field(
        description="Additional searches if more research is required."
    )

