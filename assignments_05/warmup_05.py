"""Week 5 experiments. Run with a root .env containing OPENAI_API_KEY."""
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / '.env')


def run_warmups():
    client = OpenAI()

    def complete(prompt, **kwargs):
        return client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': prompt}], **kwargs
        )

    def show(label, response):
        print(f'\n{label}: {response.choices[0].message.content}')

    # --- Completions API ---
    # API Q1
    response = complete('What is one thing that makes Python a good language for beginners?')
    show('API Q1 response', response)
    print('API Q1 model:', response.model)
    print('API Q1 total tokens:', response.usage.total_tokens)

    # API Q2
    temperatures = [0, 0.7, 1.5]
    for temperature in temperatures:
        show(f'API Q2 temperature={temperature}', complete(
            'Suggest a creative name for a data engineering consultancy.',
            temperature=temperature))
    # Observed: 0 and 0.7 both returned DataForge Solutions; 1.5 returned
    # Data Forge Consultants. Higher temperature did not guarantee much variation.
    # I would choose 0 for consistency, although it does not guarantee identical output.
    # This single run illustrates that sampling settings do not guarantee diversity.

    # API Q3
    response = complete('Give me a one-sentence fun fact about pandas (the animal, not the library).',
                        n=3, temperature=1.0)
    for number, choice in enumerate(response.choices, 1):
        print(f'API Q3 completion {number}: {choice.message.content}')

    # API Q4
    show('API Q4 max_tokens=15', complete('Explain how neural networks work.', max_tokens=15))
    # The small output budget can cut the explanation off mid-sentence. A token
    # limit controls cost and response length, but does not ensure a complete answer.

    # --- System Messages and Personas ---
    # System Q1
    personas = [
        'You are a patient, encouraging Python tutor. Explain simply and end with encouragement.',
        'You are a terse technical reference manual. Give a definition and one code example.'
    ]
    for number, persona in enumerate(personas, 1):
        response = client.chat.completions.create(model='gpt-4o-mini', messages=[
            {'role': 'system', 'content': persona},
            {'role': 'user', 'content': "I don't understand what a list comprehension is."}])
        show(f'System Q1 persona {number}', response)
    # Observed: the tutor gave a longer walkthrough and encouragement; the manual
    # returned a short definition and a single code example.

    # System Q2: history is constructed manually, without a loop or user input.
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': "My name is Jordan and I'm learning Python."},
        {'role': 'assistant', 'content': 'Nice to meet you, Jordan! Python is a great choice. What would you like to work on?'},
        {'role': 'user', 'content': 'Can you remind me what my name is?'}]
    show('System Q2 memory', client.chat.completions.create(model='gpt-4o-mini', messages=messages))
    # Jordan's name is in the history sent with this request; no server-side memory is needed.

    # --- Prompt Engineering ---
    reviews = [
        'The onboarding process was smooth and the team was welcoming.',
        'The software crashes constantly and support never responds.',
        'Great price, but the documentation is nearly impossible to follow.']
    # Prompt Q1 — Zero-Shot
    # Prompt Q2 — One-Shot
    # Prompt Q3 — Few-Shot
    examples = [
        '',
        'Example:\nReview: Fast shipping but the item arrived damaged.\nSentiment: mixed\n',
        'Examples:\nReview: Friendly staff and excellent service.\nSentiment: positive\n'
        'Review: Nothing worked and nobody helped.\nSentiment: negative\n'
        'Review: Fast shipping but the item arrived damaged.\nSentiment: mixed\n']
    for question, example in enumerate(examples, 1):
        for number, review in enumerate(reviews, 1):
            show(f'Prompt Q{question} review {number}', complete(
                'Classify the review as positive, negative, or mixed.\n'
                + example + f'Review: {review}\nSentiment:'))
    # Observed: zero-shot labels were capitalized; one-shot and few-shot labels
    # were lowercase. All three approaches agreed on the classifications.
    # Zero-shot is economical for familiar tasks. One-shot demonstrates a format.
    # Few-shot shows category boundaries and is useful for ambiguous classifications.

    # Prompt Q4 — Chain of Thought
    show('Prompt Q4 salary calculation', complete(
        'A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later '
        'takes a new job that pays $7,500 more per year than her post-raise salary. '
        'What is her final annual salary? Show the calculation step by step, '
        'then label the final answer clearly.'))
    # Explicit intermediate calculations help keep the percentage base and order
    # of operations straight. Check independently: 85000 * 1.12 + 7500 = 102700.

    # Prompt Q5 — Structured Output
    review = ("I've been using this tool for three months. It handles large datasets well, "
              'but the UI is clunky and the export options are limited.')
    response = complete('Analyze this review. Return ONLY valid JSON with keys sentiment, '
                        'confidence (a float from 0 to 1), and reason (one sentence).\n' + review,
                        response_format={'type': 'json_object'})
    raw = response.choices[0].message.content
    print('Prompt Q5 raw response:', raw)
    try:
        parsed = json.loads(raw)
        for field in ('sentiment', 'confidence', 'reason'):
            print(f'Prompt Q5 {field}: {parsed[field]}')
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        print('Prompt Q5 parsing failed:', error)
        print('Prompt Q5 raw response for debugging:', raw)

    # Prompt Q6 — Delimiters
    passages = [
        'First boil a pot of water. Once boiling, add a handful of salt and the '
        'pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice.',
        'The afternoon sunlight filled the kitchen. The garden was quiet.']
    for number, user_text in enumerate(passages, 1):
        response = complete(f'''You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: No steps provided.

```{user_text}```''')
        show(f'Prompt Q6 passage {number}', response)
        if number == 2:
            print('Prompt Q6 exact fallback confirmed:',
                  response.choices[0].message.content.strip() == 'No steps provided.')
    # Delimiters distinguish source text from instructions; they are not a complete
    # defense against malicious instructions embedded in that source text.

    # --- Local Models with Ollama ---
    # Ollama Q1
    # Run: ollama run qwen3:0.6b "Explain what a large language model is in two sentences."
    # Actual Ollama final answer (verbatim; CLI thinking preamble omitted):
    # A large language model is an advanced artificial intelligence system trained on
    # extensive textual data to understand and generate human-like language, enabling tasks
    # such as writing, answering questions, and more. It uses complex algorithms and massive
    # datasets to learn patterns and context, making it highly versatile for various
    # applications.
    show('Ollama Q1 OpenAI comparison', complete(
        'Explain what a large language model is in two sentences.'))
    # Observed locally: Qwen returned two fairly technical sentences, mentioning
    # training data, language generation, and pattern learning. The CLI also printed
    # a thinking preamble, so its full output exceeded the requested two sentences.
    # Observed OpenAI comparison: both described training on extensive text data
    # and generating language. OpenAI explicitly mentioned neural networks and
    # translation; Qwen used broader wording about algorithms and applications.
    # OpenAI returned only two sentences, without the visible thinking preamble.
    # Advantage: local prompts can stay on this machine and inference works offline
    # after download. Disadvantage: model quality/speed depend on local hardware.


if __name__ == '__main__':
    run_warmups()
