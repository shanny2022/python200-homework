"""Week 6 mini-project: Groundwork Coffee Co. document Q&A assistant."""
from pathlib import Path
import os

from dotenv import load_dotenv


def main():
    repo = Path(__file__).resolve().parents[1]
    load_dotenv(repo / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY was not found in the repository .env file.")
    else:
        print("API key loaded successfully.")

    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    docs_dir = Path(os.environ.get(
        "GROUNDWORK_DOCS_DIR",
        repo / "lessons/06_AI_augmentation/resources/groundwork_docs",
    )).expanduser()
    assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

    documents = SimpleDirectoryReader(str(docs_dir)).load_data()
    print(f"Loaded {len(documents)} document pages.")
    for document in documents:
        print("Document:", document.metadata.get("file_name", "(unknown file)"))

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini")
    index = VectorStoreIndex.from_documents(documents)
    engine = index.as_query_engine(similarity_top_k=3)
    print("Index built successfully. Ready to answer questions.")

    questions = [
        "What are Groundwork's hours on weekends?",
        "Do you offer any dairy-free milk options?",
        "How does the loyalty program work?",
        "How did Groundwork Coffee get started?",
        "Do you offer catering or wholesale orders?",
    ]
    for question in questions:
        response = engine.query(question)
        print(f"\nQuestion: {question}\nAnswer: {response}")
        if response.source_nodes:
            source = response.source_nodes[0]
            print("Top source:", source.node.metadata.get("file_name", "(unknown file)"))
            print("Similarity:", source.score)
            print("Chunk:", source.node.get_content()[:200])
        else:
            print("Top source: no source nodes returned")
    # Observed with the supplied Groundwork files: four of five answers were
    # supported by the FAQ, menu, story, and catering material. The dairy-free
    # answer correctly named oat, almond, and soy milk. The weekend-hours answer
    # failed: retrieval ranked our_story.txt first (0.534) and omitted the FAQ
    # from all three top chunks, so the model incorrectly said hours were absent.
    # Its wording sounded definite despite the missed evidence. The catering
    # answer was correct but brief, omitting details such as the 72-hour notice.

    # Step 5 — deliberately out-of-scope question to probe abstention.
    failure_query = "What is the shop manager's home address?"
    response = engine.query(failure_query)
    print(f"\nFailure probe: {failure_query}\nFull response: {response}")
    for number, source in enumerate(response.source_nodes, 1):
        print(f"Source {number}: {source.node.metadata.get('file_name', '(unknown file)')}")
        print("Similarity:", source.score)
        print("Chunk:", source.node.get_content()[:200])
    # Observed: the address is absent. Retrieval returned FAQ, catering, and
    # story chunks (scores 0.284, 0.256, and 0.239); the FAQ includes business
    # hours but no home address. The model explicitly said the address was not
    # provided and did not guess. This run abstained safely, though its top
    # chunks were only loosely related. A relevance threshold and explicit
    # abstention instruction could make that behavior more reliable.


if __name__ == "__main__":
    main()


# --- Step 6: Reflection ---
# The project uses 34 nonblank, noncomment lines for document setup, indexing,
# the five-query loop, and source reporting. LlamaIndex handles chunking,
# embedding calls, vector storage, similarity search, and prompt assembly; a
# manual keyword pipeline needs more code for retrieval and prompt construction.
# Frameworks make these standard pieces quick to compose, while still requiring
# us to inspect retrieval quality and decide how the model should handle gaps.
#
# Another valuable use case is an employee support assistant over HR policies:
# staff could ask about leave, benefits, or onboarding and receive answers linked
# to current internal documents.
#
# RAG cannot fully prevent a model from misreading relevant passages or adding an
# unsupported detail. Retrieval supplies evidence, but generation and evaluation
# can still fail.
