from graph.state import ResearchState

from llm.client import invoke_llm

from prompts.templates import REVIEWER_PROMPT

from tools.pdf_tools import parse_paper_pdf
from rag.faiss_index import build_index
from tools.retrieval_tools import retrieve_context


def reviewer_node(state: ResearchState):

    topic = state["research_topic"]

    papers = state["papers"]

    # Read Papers
    for paper in papers:

        pdf_url = paper.get("pdf_url")

        if not pdf_url:
            continue

        text = parse_paper_pdf.invoke(
            {
                "pdf_url": pdf_url
            }
        )

        build_index(
            text=text,
            paper_title=paper.get(
                "title",
                "Unknown"
            ),
            source=pdf_url
        )

    # Retrieve Context
    context = retrieve_context.invoke(
        {
            "query": topic
        }
    )

    prompt = REVIEWER_PROMPT.format(
        topic=topic,
        context=context
    )

    review = invoke_llm(prompt)

    return {
        "summaries": [review],
        "research_gaps": review
    }