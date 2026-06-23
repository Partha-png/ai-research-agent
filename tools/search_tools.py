from langchain.tools import tool
import arxiv


@tool
def search_papers(query: str):
    """Search for academic papers on arXiv based on a query string."""

    search = arxiv.Search(
        query=query,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    client = arxiv.Client()

    for paper in client.results(search):
        papers.append({
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "summary": paper.summary[:300],
            "pdf_url": paper.pdf_url
        })

    return papers

if __name__ == "__main__":

    papers = search_papers.invoke({"query": "stock market prediction transformers"})
    print(papers)