from data.paragraph_to_sentence_list import paragraph_to_sentence_list


class DocAbstractToSentenceListTransform:
    """Rotate by one of the given angles."""

    def __init__(self):
        pass

    def __call__(self, x):
        x['ABSTRACT'] = paragraph_to_sentence_list(x['ABSTRACT'])
        return x