from typing import List

import numpy as np

from allennlp import data as allen_data


@allen_data.BatchSampler.register("token_count")
class TokenCountBatchSampler(allen_data.BatchSampler):

    def __init__(self, dataset, word_batch_size: int = 2500, shuffle_dataset: bool = True):
        self._index = 0
        self.shuffle_dataset = shuffle_dataset
        self.batch_dataset = self._batchify(dataset, word_batch_size)
        if shuffle_dataset:
            self._shuffle()

    @staticmethod
    def _batchify(dataset, word_batch_size) -> List[List[int]]:
        dataset = list(dataset)
        batches = []
        batch = []
        words_count = 0
        lengths = [len(instance.fields["sentence"].tokens) for instance in dataset]
        argsorted_lengths = np.argsort(lengths)
        for idx in argsorted_lengths:
            words_count += lengths[idx]
            batch.append(idx)
            if words_count > word_batch_size:
                batches.append(batch)
                words_count = 0
                batch = []
        return batches

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self.batch_dataset):
            if self.shuffle_dataset:
                self._index = 0
                self._shuffle()
            raise StopIteration()

        batch = self.batch_dataset[self._index]
        self._index += 1
        return batch

    def _shuffle(self):
        indices = np.random.permutation(range(len(self.batch_dataset)))
        self.batch_dataset = np.array(self.batch_dataset)[indices].tolist()

    def __len__(self):
        return len(self.batch_dataset)
