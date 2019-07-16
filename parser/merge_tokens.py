from typing import List, Dict, Optional

import numpy as np

from lab.logger import Logger
from parser import tokenizer
from parser.load import EncodedFile


class InputProcessor:
    """
    TODO: We should do this at tokenizer level
    """
    TYPE_MASK_BASE = 1 << 20
    MAX_LENGTH = [1, 80, 25]

    def __init__(self, logger: Logger):
        self.logger = logger

        self.infos: List[List[IdentifierInfo]] = [[], []]
        self.dictionaries: List[Dict[str, int]] = [{} for _ in self.infos]
        self.arrays: List[np.ndarray] = [np.array([], dtype=np.uint8) for _ in self.infos]
        self.counts: List[int] = [0 for _ in self.infos]
        types = [tokenizer.TokenType.name, tokenizer.TokenType.number]
        # -1 because this is used right now for decoding,
        # and we've added 1 since 0 is used for padding
        self.offsets: List[int] = [0] + [tokenizer.get_vocab_offset(t) - 1 for t in types]

    def _add_to(self, type_idx: int, key: str, arr: np.ndarray):
        idx = self.dictionaries[type_idx]
        infos: List[IdentifierInfo] = self.infos[type_idx]
        data_array = self.arrays[type_idx]

        if key in idx:
            infos[idx[key]].count += 1
            return

        idx[key] = len(infos)
        infos.append(IdentifierInfo(len(infos), len(data_array), len(arr), key))

        self.arrays[type_idx] = np.concatenate((data_array, arr), axis=0)

    def gather(self, input_codes: np.ndarray):
        types = [tokenizer.TokenType.name, tokenizer.TokenType.number]
        offsets: List[int] = [tokenizer.get_vocab_offset(t) for t in types]
        strings: List[Optional[str]] = [None for _ in types]
        arrays: List[List[int]] = [[] for _ in types]

        for c in input_codes:
            t = tokenizer.DESERIALIZE[c]
            for type_idx, token_type in enumerate(types):
                if t.type != token_type:
                    if strings[type_idx] is not None:
                        self._add_to(type_idx, strings[type_idx],
                                     np.array(arrays[type_idx], dtype=np.uint8))
                        strings[type_idx] = None
                        arrays[type_idx] = []
                else:
                    ch = tokenizer.DECODE[c][0]
                    # add one because 0 is for padding
                    arrays[type_idx].append(c + 1 - offsets[type_idx])
                    if strings[type_idx] is None:
                        strings[type_idx] = ch
                    else:
                        strings[type_idx] += ch

        for type_idx, _ in enumerate(types):
            if strings[type_idx] is not None:
                self._add_to(type_idx, strings[type_idx],
                             np.array(arrays[type_idx], dtype=np.uint8))

    def gather_files(self, files: List[EncodedFile]):
        for f in self.logger.iterator("Counting", files):
            self.gather(f.codes)

    def transform(self, input_codes: np.ndarray):
        types = [tokenizer.TokenType.name, tokenizer.TokenType.number]
        strings: List[Optional[str]] = [None for _ in types]

        type_mask = []
        codes = []

        for c in input_codes:
            t = tokenizer.DESERIALIZE[c]
            skip = False
            for type_idx, token_type in enumerate(types):
                if t.type != token_type:
                    if strings[type_idx] is not None:
                        type_mask.append(type_idx + 1)
                        idx = self.dictionaries[type_idx][strings[type_idx]]
                        codes.append(self.infos[type_idx][idx].code)
                        strings[type_idx] = None
                else:
                    ch = tokenizer.DECODE[c][0]
                    # add one because 0 is for padding
                    if strings[type_idx] is None:
                        strings[type_idx] = ch
                    else:
                        strings[type_idx] += ch

                    skip = True

            if skip:
                continue

            type_mask.append(0)
            codes.append(c)

        for type_idx, token_type in enumerate(types):
            if strings[type_idx] is not None:
                type_mask.append(type_idx + 1)
                idx = self.dictionaries[type_idx][strings[type_idx]]
                codes.append(self.infos[type_idx][idx].code)
                strings[type_idx] = None

        codes = np.array(codes, dtype=np.int32)
        type_mask = np.array(type_mask, dtype=np.int32)
        codes = type_mask * self.TYPE_MASK_BASE + codes

        return codes

    def transform_files(self, files: List[EncodedFile]) -> List[EncodedFile]:
        transformed = []
        for f in self.logger.iterator("Transforming", files):
            transformed.append(EncodedFile(f.path, self.transform(f.codes)))

        return transformed


class IdentifierInfo:
    code: int
    count: int
    offset: int
    length: int
    string: str

    def __init__(self, code, offset, length, string):
        self.code = code
        self.count = 1
        self.offset = offset
        self.length = length
        self.string = string
