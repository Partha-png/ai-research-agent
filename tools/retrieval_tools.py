from langchain.tools import tool

from rag.faiss_index import search_index


@tool
def retrieve_context(
    query: str,
    k: int = 5
):
    """
    Retrieve relevant context from indexed research papers.
    """

    results = search_index(
        query=query,
        k=k
    )

    if not results:
        return "No relevant documents found."

    context = []

    for result in results:

        context.append(
            f"""
Paper: {result['paper_title']}
Source: {result['source']}

{result['content']}
"""
        )

    return "\n\n".join(context)
if __name__ == "__main__":

    result = retrieve_context.invoke(
        {
            "query": "What is self attention?"
        }
    )

    print(result)