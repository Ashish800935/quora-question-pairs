"""
Text preprocessing for the Quora Question Pairs dataset.

Uses only Python's standard library (re) so it has no extra
dependency beyond what's already listed on the resume
(Text Preprocessing, Tokenization as Scikit-learn / NLP skills).
"""
import re

# A reasonably complete contraction map. Extend as needed.
CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot",
    "can't've": "cannot have", "'cause": "because", "could've": "could have",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will",
    "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
    "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'll": "it will", "it's": "it is", "let's": "let us",
    "ma'am": "madam", "might've": "might have", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "needn't": "need not",
    "shan't": "shall not", "she'd": "she would", "she'll": "she will",
    "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "that'd": "that would", "that's": "that is", "there'd": "there would",
    "there's": "there is", "they'd": "they would", "they'll": "they will",
    "they're": "they are", "they've": "they have", "wasn't": "was not",
    "we'd": "we would", "we'll": "we will", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is",
    "where'd": "where did", "where's": "where is", "who'll": "who will",
    "who's": "who is", "won't": "will not", "would've": "would have",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have",
}


def preprocess(question: str) -> str:
    """
    Clean a single question string.

    Steps:
      1. Lowercase + strip whitespace
      2. Replace special symbols with word equivalents (%, $, ₹, €, @)
      3. Normalize large numbers (1,000 -> 1k, 1,000,000 -> 1m ...)
      4. Expand common English contractions
      5. Strip HTML tags and residual non-alphanumeric noise
    """
    q = str(question).lower().strip()

    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')

    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    q_decontracted = []
    for word in q.split():
        word = CONTRACTIONS.get(word, word)
        q_decontracted.append(word)
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have").replace("n't", " not")
    q = q.replace("'re", " are").replace("'ll", " will")

    # Strip HTML tags (defensive, some Quora questions had markup)
    q = re.sub(r'<.*?>', '', q)

    # Remove punctuation except apostrophes that survived, keep digits/letters
    q = re.sub(r'[^a-z0-9\s]', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()

    return q


def preprocess_series(series):
    """Vectorized helper to preprocess a pandas Series of questions."""
    return series.apply(preprocess)
