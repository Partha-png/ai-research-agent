from graph.state import ResearchState

from llm.client import invoke_llm

from prompts.templates import WRITING_PROMPT


def writing_node(
    state: ResearchState
):

    topic = state["research_topic"]

    summaries = state["summaries"]
    gaps = state["research_gaps"]
    methodology = state["methodology"]
    references = state["references"]

    summary_text = "\n\n".join(summaries)
    reference_text = "\n".join(references)

    prompt = WRITING_PROMPT.format(
        topic=topic,
        summary=summary_text,
        gaps=gaps,
        methodology=methodology,
        references=reference_text
    )

    final_draft = invoke_llm(
        prompt
    )

    return {
        "final_draft": final_draft
    }
