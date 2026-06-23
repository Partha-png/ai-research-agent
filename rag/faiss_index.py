from pathlib import Path
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "faiss_db"



vector_store = None



def build_index(
    text: str,
    paper_title: str = "Unknown Paper",
    source: str = "Unknown Source"
):
    """
    Creates or updates the FAISS index from paper text.
    Uses section-aware chunking.
    """

    global vector_store

    section_pattern = re.compile(
        r"(?im)^(abstract|introduction|related work|literature review|background|methodology|methods|experiments|results|discussion|conclusion|references)\s*$"
    )

    matches = list(section_pattern.finditer(text))

    chunks = []
    chunk_sections = []


    if not matches:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = splitter.split_text(text)

        chunk_sections = [
            "Unknown Section"
            for _ in chunks
        ]



    else:

        for i, match in enumerate(matches):

            section_name = match.group(1).title()

            start = match.start()

            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(text)
            )

            section_text = text[start:end]

            # Small section
            if len(section_text) < 3000:

                chunks.append(section_text)
                chunk_sections.append(section_name)

            # Large section
            else:

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150,
                    separators=["\n\n", "\n", ".", " ", ""]
                )

                sub_chunks = splitter.split_text(
                    section_text
                )

                chunks.extend(sub_chunks)

                chunk_sections.extend(
                    [section_name] * len(sub_chunks)
                )



    metadatas = []

    for idx in range(len(chunks)):

        metadatas.append(
            {
                "paper_title": paper_title,
                "section": chunk_sections[idx],
                "chunk_id": idx,
                "source": source
            }
        )


    if vector_store is None:

        vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=embedding_model,
            metadatas=metadatas
        )

    else:

        vector_store.add_texts(
            texts=chunks,
            metadatas=metadatas
        )

    return len(chunks)



def search_index(
    query: str,
    k: int = 5
):
    """
    Semantic search over indexed papers.
    Returns top-k matching chunks.
    """

    global vector_store

    if vector_store is None:
        return []

    docs_with_distances = (
        vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
    )

    results = []

    for doc, distance in docs_with_distances:

        results.append(
            {
                "content": doc.page_content,
                "distance": float(distance),
                "paper_title": doc.metadata.get(
                    "paper_title",
                    "Unknown Paper"
                ),
                "section": doc.metadata.get(
                    "section",
                    "Unknown Section"
                ),
                "chunk_id": doc.metadata.get(
                    "chunk_id",
                    -1
                ),
                "source": doc.metadata.get(
                    "source",
                    "Unknown Source"
                )
            }
        )

    return results



def save_index():
    """
    Save FAISS index locally.
    """

    global vector_store

    if vector_store is None:
        return

    vector_store.save_local(
        str(INDEX_PATH)
    )




def load_index():
    """
    Load FAISS index from disk.
    """

    global vector_store

    if not INDEX_PATH.exists():
        return False

    vector_store = FAISS.load_local(
        str(INDEX_PATH),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    return True




def has_index():
    """
    Returns True if an index is loaded.
    """

    return vector_store is not None




if __name__ == "__main__":

    sample_text = """
    ABSTRACT

    Transformers are deep learning models
    that use self-attention mechanisms.

    INTRODUCTION

    They have become the foundation
    of modern NLP systems.

    METHODOLOGY

    The architecture was introduced
    in the paper Attention Is All You Need.

    Transformers outperform RNNs and LSTMs
    on many NLP tasks.

    RESULTS

    Self-attention allows the model
    to capture long-range dependencies.
    """

    chunks_created = build_index(
        text=sample_text,
        paper_title="Attention Is All You Need",
        source="sample.pdf"
    )

    print(f"\nCreated {chunks_created} chunks")

    results = search_index(
        query="What is self attention?"
    )

    print("\nSearch Results")

    for idx, result in enumerate(results, start=1):

        print("\n" + "=" * 60)

        print(f"Result #{idx}")
        print(f"Paper     : {result['paper_title']}")
        print(f"Section   : {result['section']}")
        print(f"Source    : {result['source']}")
        print(f"Chunk ID  : {result['chunk_id']}")
        print(f"Distance  : {result['distance']}")

        print("\nContent:")
        print(result["content"][:300])

    save_index()

    print("\nFAISS index saved successfully.")

    loaded = load_index()

    if loaded:
        print("FAISS index loaded successfully.")