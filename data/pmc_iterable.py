from math import ceil

import ir_datasets
import pickle
from torch.utils.data import IterableDataset

class PMCIterable(IterableDataset):
    def __init__(self, labeled_ids_or_filename=(), train=False, test=False, train_percentage=0.8, format='default', transform=None):
        if train and test:
            raise ValueError("train and test cannot both be True")
        if type(labeled_ids_or_filename) == str:
            self.labeled_ids = pickle.load(open(labeled_ids_or_filename, "rb"))
        else:
            self.labeled_ids = labeled_ids_or_filename
        self.train = train
        self.test = test
        self.train_percentage = train_percentage
        self.loaded_object = ir_datasets.load("pmc/v2/trec-cds-2016")
        self.format = format  # only 'aspire', 'default' or 'beep' are supported
        self.transform = transform

    def __iter__(self):
        if len(self.labeled_ids) > 0:
            if self.train:
                self._iterator = self.loaded_object.docs_store().get_many(self.labeled_ids).values()
                self._iterator = (x for i, x in enumerate(self._iterator) if
                                  i < len(self.labeled_ids) * self.train_percentage)

            elif self.test:
                self._iterator = self.loaded_object.docs_store().get_many(self.labeled_ids).values()
                self._iterator = (x for i, x in enumerate(self._iterator) if
                                  i >= len(self.labeled_ids) * self.train_percentage)

            else:
                self._iterator = self.loaded_object.docs_store().get_many(self.labeled_ids).values().__iter__()
        else:
            if self.train:
                self._iterator = self.loaded_object.docs_iter()[:self.train_percentage]

            elif self.test:
                self._iterator = self.loaded_object.docs_iter()[self.train_percentage:]

            else:
                self._iterator = self.loaded_object.docs_iter()

        return self

    def __next__(self):
        doc = next(self._iterator)
        if self.format == 'beep':
            ret_val = {doc.doc_id: {'text': doc.body, 'year': -1000}}
        elif self.format == 'aspire':
            ret_val = {'ID': doc.doc_id, 'TITLE': doc.title, 'ABSTRACT': doc.abstract if doc.abstract else doc.body}
        elif self.format == 'default':
            ret_val = doc
        else:
            raise ValueError("format must be either 'beep', 'aspire' or 'default'")

        if self.transform:
            ret_val = self.transform(ret_val)
        return ret_val

    def __len__(self):
        if self.labeled_ids and self.train:
            return ceil(len(self.labeled_ids) * self.train_percentage)
        elif self.labeled_ids and self.test:
            return ceil(len(self.labeled_ids) * (1 - self.train_percentage))
        elif self.labeled_ids:
            return len(self.labeled_ids)
        elif self.train:
            return ceil(len(self.loaded_object.docs_iter()) * self.train_percentage)
        elif self.test:
            return ceil(
                len(self.loaded_object.docs_iter()) - ceil(len(self.loaded_object.docs_iter()) * self.train_percentage))
        else:
            return len(self.loaded_object.docs_iter())


if __name__ == '__main__':
    p = PMCIterable(labeled_ids_or_filename=('4504085', '4504086', '4504087', '4504088', '4504089'), train=True,
                    format='aspire')
    for x in p:
        print(x['ID'])
