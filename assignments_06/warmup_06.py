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
    # Observed: loyalty.txt wins, unexpectedly, because the function leaves
    # "your" in the query tokens and that word appears in "your choice" there.
    # It misses hours.txt because "weekends" and "weekend" are not normalized.

    # Keyword Question 2
    query = "Do you have anything without caffeine?"
    result = simple_keyword_retrieval(query, documents, verbose=True)
    print("Keyword Q2 selected document:", result[0][0])
    # Observed: no document was selected. The query's wording shares no tokens
    # with these four short documents, so keyword overlap misses the menu. A
    # semantic or synonym-aware search could connect caffeine-free with decaf.

    # Keyword Question 3 — prediction before running retrieval:
    # Prediction: loyalty.txt, because rewards and loyalty are related ideas.
    query = "How do I sign up for rewards?"
    result = simple_keyword_retrieval(query, documents, verbose=True)
    print("Keyword Q3 selected document:", result[0][0])
    # Observed: no match. "Rewards" does not occur in loyalty.txt, and the
    # retriever does not know that rewards and loyalty are synonyms; prediction
    # was wrong, demonstrating the exact-vocabulary limitation.


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
    docs_dir = Path(os.environ.get(
        "BRIGHTLEAF_DOCS_DIR",
        repo / "lessons/06_AI_augmentation/resources/brightleaf_pdfs",
    )).expanduser()
    assert docs_dir.exists(), f"BrightLeaf document directory not found: {docs_dir}"
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    # LlamaIndex 0.14 defaults to a retired model; use the current course model.
    Settings.llm = OpenAI(model="gpt-4o-mini")
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
    # Observed with the supplied PDFs: the leading chunk matched each query and
    # both answers were direct and specific. For benefits, however, the other
    # two top-three chunks were partnerships and mission content; for security,
    # benefits and mission passages were also retrieved unexpectedly.

    # LlamaIndex Question 2
    q = questions[0]
    for top_k in (1, 5):
        response = index.as_query_engine(similarity_top_k=top_k).query(q)
        print(f"\nTop-k={top_k} answer: {response}")
        for node in response.source_nodes:
            print(f"Score: {node.score}; chunk: {node.node.get_content()[:150]}")
    # Observed: top-k=1 and top-k=5 both gave a correct benefits summary with a
    # similar direct tone; top-k=5 added some useful detail but also included
    # unrelated chunks. More context did not materially improve this answer and
    # could distract the model on a less focused query.

    # LlamaIndex Question 3
    difficult_query = "What is BrightLeaf's favorite coffee shop?"
    response = engine.query(difficult_query)
    print(f"\nDifficult question: {difficult_query}\nAnswer: {response}")
    for node in response.source_nodes:
        print(f"Score: {node.score}; chunk: {node.node.get_content()}")
    # Observed: retrieval returned unrelated mission, partnership, and benefits
    # passages (top score 0.467). The model correctly said the documents did not
    # give a favorite coffee shop, in a clear, direct tone. Both evaluators later
    # scored this abstention 0.0, so judge feedback needs interpretation. A
    # similarity threshold and explicit abstention policy could reinforce this.

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
    # question. Here the in-domain response scored 1.0 for both, while the
    # abstention to the out-of-domain question scored 0.0 for both. The judge
    # appears to penalize the abstention because it neither cites support nor
    # gives the requested favorite shop. LLM-as-a-judge helps assess nuanced
    # answers where exact matching is inadequate, but scores are estimates and
    # should be read alongside the response and retrieved evidence.


if __name__ == "__main__":
    run_keyword_questions()
    run_llamaindex_questions()
