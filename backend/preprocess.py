import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  # Instantiate fresh—no need to save/load

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# No load needed—create stemmer instance directly (matches your training)
stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    """
    Clean and stem text to match training pipeline.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords (as in your imports)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)