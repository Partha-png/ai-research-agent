from langgraph.graph import StateGraph, END

from graph.state import ResearchState

from graph.nodes.discovery_node import discovery_node
from graph.nodes.reviewer_node import reviewer_node
from graph.nodes.methodology_node import methodology_node
from graph.nodes.citation_node import citation_node
from graph.nodes.writing_node import writing_node


def build_graph():

    graph = StateGraph(ResearchState)

    # Nodes
    graph.add_node(
        "discovery",
        discovery_node
    )

    graph.add_node(
        "reviewer",
        reviewer_node
    )

    graph.add_node(
        "methodology",
        methodology_node
    )

    graph.add_node(
        "citation",
        citation_node
    )

    graph.add_node(
        "writing",
        writing_node
    )

    # Flow
    graph.set_entry_point(
        "discovery"
    )

    graph.add_edge(
        "discovery",
        "reviewer"
    )

    graph.add_edge(
        "reviewer",
        "methodology"
    )

    graph.add_edge(
        "methodology",
        "citation"
    )

    graph.add_edge(
        "citation",
        "writing"
    )

    graph.add_edge(
        "writing",
        END
    )

    return graph.compile()
if __name__ == "__main__":

    app = build_graph()

    result = app.invoke(
        {
            "research_topic":
            "Transformer based stock market forecasting"
        }
    )

    print(result)