from graph.state import ResearchState

from llm.client import invoke_llm

from prompts.templates import (
    METHODOLOGY_PROMPT
)


def methodology_node(
    state: ResearchState
):

    topic = state["research_topic"]

    gaps = state["research_gaps"]

    prompt = METHODOLOGY_PROMPT.format(
        topic=topic,
        gaps=gaps
    )

    methodology = invoke_llm(
        prompt
    )

    return {
        "methodology": methodology
    }