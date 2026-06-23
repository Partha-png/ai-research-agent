import re

from graph.state import ResearchState
from llm.client import invoke_llm
from prompts.templates import DISCOVERY_PROMPT

from tools.search_tools import search_papers


def _sanitize_query(raw: str) -> str | None:
    """
    Strip markdown formatting and reject lines that look like
    headers, prose, or bullet points rather than search queries.
    Returns None if the line should be skipped entirely.
    """
    # Remove markdown bold/italic asterisks and hashes
    cleaned = re.sub(r"[*#`]", "", raw).strip()

    # Remove leading list/numbering markers like "1.", "-", "•"
    cleaned = re.sub(r"^[\d]+[.)]\s*|^[-•]\s*", "", cleaned).strip()

    # Skip lines that are clearly headers or labels (contain a colon mid-line)
    if re.search(r":\s", cleaned):
        return None

    # Skip very short or very long strings
    if len(cleaned) < 5 or len(cleaned) > 120:
        return None

    # Cap at 100 characters to stay safely within arxiv query limits
    return cleaned[:100]


def discovery_node(state: ResearchState):

    topic = state["research_topic"]

    prompt = DISCOVERY_PROMPT.format(
        topic=topic
    )

    queries = invoke_llm(prompt)

    raw_lines = [q.strip() for q in queries.split("\n") if q.strip()]

    # Sanitize every line; drop anything that doesn't look like a real query
    query_list = [
        sanitized
        for raw in raw_lines
        if (sanitized := _sanitize_query(raw)) is not None
    ]

    all_papers = []

    for query in query_list:

        papers = search_papers.invoke(
            {"query": query}
        )

        if isinstance(papers, list):
            all_papers.extend(papers)

    return {
        "papers": all_papers
    }