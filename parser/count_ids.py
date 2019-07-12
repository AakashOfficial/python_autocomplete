from typing import Dict, Optional, List

import parser.load
from lab.logger import Logger
from parser import tokenizer
import numpy as np

logger = Logger()


class Stats:
    def __init__(self):
        self.identifiers: Dict[str, int] = {}
        self.numbers: Dict[str, int] = {}

    def add_identifier(self, identifier: str):
        if identifier in self.identifiers:
            self.identifiers[identifier] += 1
        else:
            self.identifiers[identifier] = 1
            if len(self.identifiers) % 10000 == 0:
                logger.info(ids=len(self.identifiers))

    def add_number(self, number: str):
        if number in self.numbers:
            self.numbers[number] += 1
        else:
            self.numbers[number] = 1
            if len(self.numbers) % 10000 == 0:
                logger.info(nums=len(self.numbers))

    def count_file(self, file: parser.load.EncodedFile):
        identifier: Optional[str] = None
        number: Optional[str] = None

        for c in file.codes:
            t = tokenizer.DESERIALIZE[c]
            if t.type != tokenizer.TokenType.name:
                if identifier is not None:
                    self.add_identifier(identifier)
                    identifier = None
            else:
                ch = tokenizer.DECODE[c][0]
                if identifier is None:
                    identifier = ch
                else:
                    identifier += ch

            if t.type != tokenizer.TokenType.number:
                if number is not None:
                    self.add_number(number)
                    number = None
            else:
                ch = tokenizer.DECODE[c][0]
                if number is None:
                    number = ch
                else:
                    number += ch

    def count_files(self, files: List[parser.load.EncodedFile]):
        with logger.section("Counting", total_steps=len(files)):
            for i, f in enumerate(files):
                self.count_file(f)
                logger.progress(i + 1)


def main():
    files = parser.load.load_files()
    stats = Stats()
    stats.count_files(files)
    logger.info(ids=len(stats.identifiers), nums=len(stats.numbers))
    # There are 1714343 ids that were present at least twice
    # 946176 at least 4 time
    # 350128 at least 11 times

    # 222994 ids and 532812 numbers

    logger.info(len_id=np.max([len(k) for k in stats.identifiers.keys()]),
                len_num=np.max([len(k) for k in stats.numbers.keys()]))


if __name__ == '__main__':
    main()
