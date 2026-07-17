"""
SmartReview — Prétraitement NLP
Master BD & IA | AIDI Omaima | ELMOUSSAOUI Fatima

Prétraitement sans dépendance réseau NLTK (stopwords embarqués + lemmatisation légère).
"""

from __future__ import annotations

import re

# Stopwords anglais (liste NLTK classique, embarquée pour reproductibilité offline)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're",
    "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he",
    "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's",
    "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "don't", "should", "should've", "now", "d", "ll",
    "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn",
    "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven",
    "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't",
    "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't",
    "weren", "weren't", "won", "won't", "wouldn", "wouldn't", "br",
}


def simple_lemmatize(token: str) -> str:
    """Lemmatisation morphologique légère (règles anglaises courantes)."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if len(stem) > 2:
            return stem
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("ly") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def clean_text(text: str) -> str:
    """Nettoyage basique : minuscules, HTML, ponctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_full(text: str) -> str:
    """
    Prétraitement NLP complet (H2) :
    nettoyage → tokenisation → stopwords → lemmatisation.
    """
    text = clean_text(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [simple_lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_minimal(text: str) -> str:
    """Sans prétraitement avancé (condition contrôle H2) : nettoyage seul."""
    return clean_text(text)
