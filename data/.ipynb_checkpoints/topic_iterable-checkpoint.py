from math import ceil
import random
import pickle
from torch.utils.data import IterableDataset

class TopicIterable(IterableDataset):
    def __init__(self, topic_file_name, train=False, test=False, train_percentage=0.8, format='default', seed=42, transform=None):
        if train and test:
            raise ValueError("train and test cannot both be True")
        self.topics = pickle.load(open(topic_file_name, "rb"))
        random.seed(seed)
        random.shuffle(self.topics)
        self.train = train
        self.test = test
        self.train_percentage = train_percentage
        self.format = format  # only 'aspire', 'default' or 'beep' are supported
        self.transform = transform

    def __iter__(self):
        if self.train:
            self._iterator = iter(self.topics[:ceil(self.train_percentage * len(self.topics))])

        elif self.test:
            self._iterator = iter(self.topics[ceil(self.train_percentage * len(self.topics)):])

        else:
            self._iterator = iter(self.topics)
        return self

    def __next__(self):
        if self.format == 'beep':
            doc = next(self._iterator)
        elif self.format == 'aspire':
            doc = next(self._iterator)
            type_to_co_citation_dict = {
                'diagnosis': "What is the patient's diagnosis?",
                'treatment': "How should the patient be treated?",
                'test': "What tests should the patient receive?"
            }
            doc = {'ID': doc['topic_id'], 'TITLE': doc['topic_summary'], 'ABSTRACT': doc['topic_note'],
                    'CO-CITATION-CONTEXT': type_to_co_citation_dict[doc['topic_type']]}
        elif self.format == 'default':
            doc = next(self._iterator)
        else:
            raise ValueError("format must be either 'beep', 'aspire' or 'default'")

        if self.transform:
            doc = self.transform(doc)
        return doc

    def __len__(self):
        if self.train:
            return ceil(len(self.topics) * self.train_percentage)
        elif self.test:
            return ceil(
                len(self.topics) - ceil(len(self.topics) * self.train_percentage))
        else:
            return len(self.topics)


if __name__ == '__main__':
    p = TopicIterable(topic_file_name="./data_old_format/topics.pkl", train=True, format='aspire')
    for x in p:
        print(x['CO-CITATION-CONTEXT'])
