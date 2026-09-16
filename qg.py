from transformers import pipeline

# Load the question generation model
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-small-qg-prepend")

def generate_questions_from_sentences(sentences, n_per_sentence=1):
    """
    Generate questions from a list of sentences.
    :param sentences: List of relevant sentences
    :param n_per_sentence: Number of questions to generate per sentence
    :return: List of (question, answer) pairs
    """
    qa_pairs = []

    for sentence in sentences:
        prompt = f"generate questions: {sentence}"

        # ✅ Use sampling so multiple outputs work properly
        outputs = qg_pipeline(
            prompt,
            num_return_sequences=n_per_sentence,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            max_length=64
        )

        for out in outputs:
            question_text = out['generated_text']
            qa_pairs.append((sentence, question_text))

    return qa_pairs
