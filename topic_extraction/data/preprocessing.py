import re

from topic_extraction.data.utils import STOPWORDS



def preprocess_text(document):
    # Remove all the special characters
    document = re.sub(r'/[^\w.-åæø]|_/g', ' ', str(document))

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    return document
