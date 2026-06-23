from langchain.tools import tool


@tool
def generate_references(papers: list):
    """
    Generate simple references from discovered papers.
    """

    references = []

    for idx, paper in enumerate(papers, start=1):

        title = paper.get("title", "Unknown Title")
        authors = paper.get("authors", [])
        year = paper.get("year", "Unknown")

        if authors:
            author_text = ", ".join(authors[:3])
        else:
            author_text = "Unknown Author"

        reference = (
            f"[{idx}] {author_text} ({year}). "
            f"{title}."
        )

        references.append(reference)

    return references
if __name__ == "__main__":
    import json
    with open("papers.json", "r") as f:
        papers = json.load(f)

    print(generate_references(papers))