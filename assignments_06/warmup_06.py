"""Week 6 warmups: knowledge augmentation, keyword RAG, and semantic RAG."""
import string
from pathlib import Path

from dotenv import load_dotenv
import os

# Load the repository-level .env even when this script is launched elsewhere.
if load_dotenv(Path(__file__).resolve().parents[1] / ".env"):
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- RAG Concepts ---
# Concepts Q1
# A: RAG. The internal library is large and changes quarterly, so retrieving
# current policy passages at question time avoids retraining whenever it changes.
# B: Fine-tuning. The 3,000 examples can teach a distinctive, consistent writing
# style that is not reliably available in the model's general training data.
# C: Prompt engineering/context injection. One short report fits directly in a
# prompt, making a persistent retrieval system or model training unnecessary.

# Concepts Q2
# A confidently wrong answer can be acted on before anyone thinks to verify it;
# uncertainty invites a check. For example, a confident but invented medication
# dosage could lead a patient to take a dangerous amount. Polished, decisive tone
# is often mistaken for evidence, even though it does not make a claim more true.

# Concepts Q3 — correct order and what happens at each step:
# 1. Extract text from source documents — make document contents searchable.
# 2. Split text into chunks — create manageable passages for targeted retrieval.
# 3. Convert text chunks into embeddings — represent each passage as a vector.
# 4. Receive the user's query — accept the information need.
# 5. Embed the user's query — represent the question in the same vector space.
# 6. Retrieve the most relevant chunks — select passages closest in meaning.
# 7. Inject retrieved chunks into the prompt — provide evidence to the model.
# 8. Generate a response from the LLM — answer using the supplied evidence.


# --- Keyword RAG ---
def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


def run_keyword_questions():
    documents = {
        "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
        "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
        "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
        "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
    }

    # Keyword Question 1
    query = "What are your hours on weekends?"
    result = simple_keyword_retrieval(query, documents, verbose=True)
    print("Keyword Q1 selected document:", result[0][0])
    # hours.txt wins because weekend/hours terms overlap most strongly there.

    # Keyword Question 2
    query = "Do you have anything without caffeine?"
    result = simple_keyword_retrieval(query, documents, verbose=True)
    print("Keyword Q2 selected document:", result[0][0])
    # Exact word overlap can miss a decaf product described with different terms;
    # semantic retrieval can connect related language and synonyms.

    # Keyword Question 3 — prediction before running retrieval:
    # I predict loyalty.txt: rewards and loyalty are related concepts, but this
    # retriever only compares exact words, so it may instead find no match.
    query = "How do I sign up for rewards?"
    result = simple_keyword_retrieval(query, documents, verbose=True)
    print("Keyword Q3 selected document:", result[0][0])
    # The actual result is determined by token overlap; 'rewards' does not appear
    # in the loyalty text, so a no-match result would expose the vocabulary gap.


# --- Semantic RAG Concepts ---
# Semantic Question 1
# An embedding is a list of numbers that places text in a space where passages
# with related meanings tend to be near one another. A chunk scoring 0.85 is more
# relevant than one scoring 0.30; the higher score signals greater directional
# similarity between the query and the chunk, not proof that the chunk answers it.
# Semantic search can match paraphrases because it compares learned meaning rather
# than requiring identical words.

# Semantic Question 2
# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | Query and chunk embedding vectors |
# | What is retrieved?         | Full document                     | Most similar text chunks |
# | Can it handle synonyms?    | No                                | Often, based on learned meaning |
# | Storage format             | Plain text dictionary             | Embedding vectors plus chunk text/metadata |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity between vectors |


# --- LlamaIndex ---
def run_llamaindex_questions():
    """Run the BrightLeaf exercises; requires lesson PDFs and an API key."""
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

    repo = Path(__file__).resolve().parents[1]
    docs_dir = repo / "lessons/06_AI_augmentation/resources/brightleaf_pdfs"
    assert docs_dir.exists(), f"BrightLeaf document directory not found: {docs_dir}"
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    documents = SimpleDirectoryReader(str(docs_dir)).load_data()
    print(f"Loaded {len(documents)} BrightLeaf document pages.")
    index = VectorStoreIndex.from_documents(documents)
    engine = index.as_query_engine(similarity_top_k=3)

    # LlamaIndex Question 1
    questions = ["What employee benefits does BrightLeaf offer?",
                 "What are BrightLeaf's security policies?"]
    for question in questions:
        response = engine.query(question)
        print(f"\nQuestion: {question}\nAnswer: {response}")
        for node in response.source_nodes:
            print(f"Score: {node.score}; chunk: {node.node.get_content()[:150]}")
    # Record observations after running: evaluate relevance, confidence/hedging,
    # and whether any retrieved chunk was unexpected.

    # LlamaIndex Question 2
    q = questions[0]
    for top_k in (1, 5):
        response = index.as_query_engine(similarity_top_k=top_k).query(q)
        print(f"\nTop-k={top_k} answer: {response}")
        for node in response.source_nodes:
            print(f"Score: {node.score}; chunk: {node.node.get_content()[:150]}")
    # Compare the actual responses: additional context can help, but irrelevant
    # passages can distract the model or dilute the useful evidence.

    # LlamaIndex Question 3
    difficult_query = "What is BrightLeaf's favorite coffee shop?"
    response = engine.query(difficult_query)
    print(f"\nDifficult question: {difficult_query}\nAnswer: {response}")
    for node in response.source_nodes:
        print(f"Score: {node.score}; chunk: {node.node.get_content()}")
    # This is intentionally out of domain. Inspect whether retrieval brings
    # irrelevant evidence and whether the answer abstains; a score threshold,
    # clearer abstention prompt, or better metadata filters could help.

    # LlamaIndex Question 4
    judge = OpenAI(model="gpt-4o-mini")
    faithfulness = FaithfulnessEvaluator(llm=judge)
    relevancy = RelevancyEvaluator(llm=judge)
    for q in ("What employee benefits does BrightLeaf offer?",
              "What is BrightLeaf's favorite coffee shop?"):
        response = engine.query(q)
        faith_result = faithfulness.evaluate_response(response=response)
        relevancy_result = relevancy.evaluate_response(query=q, response=response)
        print(f"\nEvaluation query: {q}")
        print("Faithfulness:", faith_result.score, faith_result.feedback)
        print("Relevancy:", relevancy_result.score, relevancy_result.feedback)
    # Faithfulness measures support in retrieved context (1 means supported,
    # 0 means unsupported); relevancy measures whether the response addresses the
    # question. LLM-as-a-judge uses a model to assess nuanced answers where exact
    # string matching is inadequate. Compare the scores above: the out-of-domain
    # query may score lower, but judge scores are estimates, not ground truth.


if __name__ == "__main__":
    run_keyword_questions()
    run_llamaindex_questions()
