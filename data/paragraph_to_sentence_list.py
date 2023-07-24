import nltk.data

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def paragraph_to_sentence_list(paragraph):
    """
    Given a paragraph, returns a list of sentences.
    :param paragraph: string
    :return: list of strings
    """
    return sent_detector.tokenize(paragraph.strip())
