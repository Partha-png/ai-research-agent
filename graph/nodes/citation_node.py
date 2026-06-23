from graph.state import ResearchState

from llm.client import invoke_llm

from prompts.templates import CITATION_PROMPT

from tools.citation_tools import generate_references


def citation_node(
    state: ResearchState
):

    papers = state["papers"]

    references = generate_references.invoke(
        {
            "papers": papers
        }
    )

    prompt = CITATION_PROMPT.format(
        references="\n".join(references)
    )

    bibliography = invoke_llm(
        prompt
    )

    return {
        "references": references,
        "bibliography": bibliography
    }
