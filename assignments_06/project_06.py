"""Week 6 mini-project: Groundwork Coffee Co. document Q&A assistant."""
from pathlib import Path

from dotenv import load_dotenv


def main():
    repo = Path(__file__).resolve().parents[1]
    load_dotenv(repo / ".env")
    if not __import__("os").getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY was not found in the repository .env file.")
    else:
        print("API key loaded successfully.")

    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding

    docs_dir = repo / "lessons/06_AI_augmentation/resources/groundwork_docs"
    assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

    documents = SimpleDirectoryReader(str(docs_dir)).load_data()
    print(f"Loaded {len(documents)} document pages.")
    for document in documents:
        print("Document:", document.metadata.get("file_name", "(unknown file)"))

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
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
    # Reflection to complete after the live run: Did the answers sound accurate
    # and well grounded? Note any surprising answer or unsupported detail.

    # Step 5 — deliberately out-of-scope question to probe abstention.
    failure_query = "What is the shop manager's home address?"
    response = engine.query(failure_query)
    print(f"\nFailure probe: {failure_query}\nFull response: {response}")
    for number, source in enumerate(response.source_nodes, 1):
        print(f"Source {number}: {source.node.metadata.get('file_name', '(unknown file)')}")
        print("Similarity:", source.score)
        print("Chunk:", source.node.get_content()[:200])
    # Review the actual retrieval and response: the requested personal detail
    # should be absent. Note whether the system retrieved unrelated material,
    # guessed confidently, or abstained. A score threshold and an explicit
    # abstention policy could reduce unsupported answers.


if __name__ == "__main__":
    main()


# --- Step 6: Reflection ---
# The core semantic pipeline here is a handful of LlamaIndex calls: configure an
# embedding model, load documents, build an index, and create a query engine.
# Implementing chunking, embedding calls, vector storage, similarity search, and
# prompt assembly by hand would require substantially more code and maintenance.
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
