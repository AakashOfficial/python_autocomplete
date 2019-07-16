from typing import List, NamedTuple, Optional

import numpy as np

import parser.load
from lab.logger import Logger
from parser import tokenizer
from parser.merge_tokens import InputProcessor, IdentifierInfo


class BatchBuilder:
    def __init__(self, input_processor: InputProcessor, logger: Logger):
        self.logger = logger
        self.processor = input_processor
        self.freqs = [self.get_frequencies(info) for info in self.processor.infos]

    @staticmethod
    def get_frequencies(info: List[IdentifierInfo]):
        freqs = [(i.code, i.count) for i in info]
        freqs.sort(reverse=True, key=lambda x: x[1])
        return [f[0] for f in freqs]

    def get_batches(self, files: List[parser.load.EncodedFile],
                    eof: int, batch_size: int, seq_len: int):
        """
        Covert raw encoded files into training/validation batches
        """

        # Shuffle the order of files
        np.random.shuffle(files)

        # Start from a random offset
        offset = np.random.randint(seq_len * batch_size)

        x_unordered = []
        y_unordered = []

        # Concatenate all the files whilst adding `eof` marker at the beginnings
        data = []
        last_clean = 0

        eof = np.array([eof], dtype=np.int32)

        for i, f in self.logger.enumerator("Get batches", files):
            if len(f.codes) == 0:
                continue

            # To make sure data type in int
            if len(data) > 0:
                data = np.concatenate((data, eof, f.codes), axis=0)
            else:
                data = np.concatenate((eof, f.codes), axis=0)
            if len(data) <= offset:
                continue
            data = data[offset:]
            offset = 0

            while len(data) >= batch_size * seq_len + 1:
                x_batch = data[:(batch_size * seq_len)]
                data = data[1:]
                y_batch = data[:(batch_size * seq_len)]
                data = data[(batch_size * seq_len):]
                if i - last_clean > 100:
                    data = np.copy(data)
                    last_clean = i

                x_batch = np.reshape(x_batch, (batch_size, seq_len))
                y_batch = np.reshape(y_batch, (batch_size, seq_len))
                x_unordered.append(x_batch)
                y_unordered.append(y_batch)

        del files
        del data

        batches = len(x_unordered)
        x = []
        y = []

        idx = [batches * i for i in range(batch_size)]
        for _ in self.logger.iterator("Order batches", batches):
            x_batch = np.zeros((batch_size, seq_len), dtype=np.int32)
            y_batch = np.zeros((batch_size, seq_len), dtype=np.int32)
            for j in range(batch_size):
                n = idx[j] // batch_size
                m = idx[j] % batch_size
                idx[j] += 1
                x_batch[j, :] = x_unordered[n][m, :]
                y_batch[j, :] = y_unordered[n][m, :]

            x_batch = np.transpose(x_batch, (1, 0))
            y_batch = np.transpose(y_batch, (1, 0))
            x.append(x_batch)
            y.append(y_batch)

        del x_unordered
        del y_unordered

        return x, y

    def create_token_array(self, token_type: int, length: int, tokens: list):
        res = np.zeros((len(tokens), length), dtype=np.uint8)

        if token_type == 0:
            res[:, 0] = tokens
            return res

        token_type -= 1
        infos = self.processor.infos[token_type]
        data_array = self.processor.arrays[token_type]

        for i, t in enumerate(tokens):
            info = infos[t]
            res[i, :info.length] = data_array[info.offset:info.offset + info.length]

        return res

    def _get_token_sets(self, x_source: np.ndarray, y_source: np.ndarray):
        seq_len, batch_size = x_source.shape

        sets: List[set] = [set() for _ in range(len(self.processor.infos) + 1)]

        for s in range(seq_len):
            for b in range(batch_size):
                type_idx = x_source[s, b] // InputProcessor.TYPE_MASK_BASE
                c = x_source[s, b] % InputProcessor.TYPE_MASK_BASE
                sets[type_idx].add(c)

                type_idx = y_source[s, b] // InputProcessor.TYPE_MASK_BASE
                c = y_source[s, b] % InputProcessor.TYPE_MASK_BASE
                sets[type_idx].add(c)

        for i in range(1, len(self.processor.infos) + 1):
            sets[i] = sets[i].union(self.freqs[i - 1][:128])

        return sets

    def build_infer_batch(self, x_source: np.ndarray):
        seq_len, batch_size = x_source.shape

        lists = [[]]
        for infos in self.processor.infos:
            lists.append([i for i in range(len(infos))])
        lists[0] = [i for i in range(tokenizer.VOCAB_SIZE)]

        dicts = [{c: i for i, c in enumerate(s)} for s in lists]

        token_data = []
        for i, length in enumerate(InputProcessor.MAX_LENGTH):
            token_data.append(self.create_token_array(i, length, lists[i]))

        x = np.zeros_like(x_source, dtype=np.int32)
        x_type = np.zeros_like(x_source, dtype=np.int8)

        for s in range(seq_len):
            for b in range(batch_size):
                type_idx = x_source[s, b] // InputProcessor.TYPE_MASK_BASE
                c = x_source[s, b] % InputProcessor.TYPE_MASK_BASE
                x[s, b] = dicts[type_idx][c]
                x_type[s, b] = type_idx

        return Batch(x, None, x_type, None, None,
                     token_data[0],
                     token_data[1],
                     token_data[2])

    def build_batch(self, x_source: np.ndarray, y_source: np.ndarray):
        seq_len, batch_size = x_source.shape

        lists = [list(s) for s in self._get_token_sets(x_source, y_source)]
        lists[0] = [i for i in range(tokenizer.VOCAB_SIZE)]

        dicts = [{c: i for i, c in enumerate(s)} for s in lists]

        token_data = []
        for i, length in enumerate(InputProcessor.MAX_LENGTH):
            token_data.append(self.create_token_array(i, length, lists[i]))

        x = np.zeros_like(x_source, dtype=np.int32)
        x_type = np.zeros_like(x_source, dtype=np.int8)
        y = np.zeros_like(y_source, dtype=np.int32)
        y_type = np.zeros_like(y_source, dtype=np.int8)
        y_idx = np.zeros_like(y_source, dtype=np.int32)

        offset = np.cumsum([0] + [len(s) for s in lists])

        for s in range(seq_len):
            for b in range(batch_size):
                type_idx = x_source[s, b] // InputProcessor.TYPE_MASK_BASE
                c = x_source[s, b] % InputProcessor.TYPE_MASK_BASE
                x[s, b] = dicts[type_idx][c]
                x_type[s, b] = type_idx

                type_idx = y_source[s, b] // InputProcessor.TYPE_MASK_BASE
                c = y_source[s, b] % InputProcessor.TYPE_MASK_BASE
                y[s, b] = dicts[type_idx][c]
                y_type[s, b] = type_idx
                y_idx[s, b] = offset[type_idx] + dicts[type_idx][c]

        return Batch(x, y, x_type, y_type, y_idx,
                     token_data[0],
                     token_data[1],
                     token_data[2])

        # return [len(s) for s in sets]

    def build_batches(self, x: List[np.ndarray], y: List[np.ndarray]):
        batches: List[Batch] = []
        for b in self.logger.iterator("Build batches", len(x)):
            batches.append(self.build_batch(x[b], y[b]))

        return batches


class Batch(NamedTuple):
    x: np.ndarray
    y: Optional[np.ndarray]
    x_type: np.ndarray
    y_type: Optional[np.ndarray]
    y_idx: Optional[np.ndarray]
    tokens: np.ndarray
    ids: np.ndarray
    nums: np.ndarray
