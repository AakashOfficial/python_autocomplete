from pathlib import Path
from typing import NamedTuple, List

import numpy as np
import math

from lab.logger import Logger
from parser import tokenizer

logger = Logger()


class EncodedFile(NamedTuple):
    path: str
    codes: np.ndarray


def _save_cache(files: List[EncodedFile]):
    packet_size = 10_000
    with logger.section("Save data", total_steps=math.ceil(len(files) / packet_size)):
        i = 1
        while len(files) > 0:
            np_object = np.array(files[:packet_size], dtype=np.object_)
            files = files[packet_size:]
            cache_path = str(Path(__file__).parent.parent / 'data' / f"all.{i}.npy")
            np.save(cache_path, np_object)
            i += 1
            logger.progress(i)


def _load_cache():
    files = []
    with logger.section("Read cache"):
        i = 1
        while True:
            try:
                cache_path = str(Path(__file__).parent.parent / 'data' / f"all.{i}.npy")

                np_object = np.load(cache_path)
                if np_object is None:
                    break
            except Exception as e:
                break

            files += [EncodedFile(f[0], f[1]) for f in np_object]
            i += 1

        if len(files) == 0:
            logger.set_successful(False)
            return None

    return files


def load_files() -> List[EncodedFile]:
    """
    Load encoded files
    """

    files = _load_cache()
    if files is not None:
        return files

    with logger.section("Read data"):
        with open(str(Path(__file__).parent.parent / 'data' / 'all.py')) as f:
            lines = f.readlines()

    with logger.section("Extract data", total_steps=len(lines)):
        files = []
        for i in range(0, len(lines), 2):
            path = lines[i][:-1]
            content = lines[i + 1][:-1]
            if content == '':
                content = []
            else:
                content = np.array([int(t) for t in content.split(' ')], dtype=np.uint8)
            files.append(EncodedFile(path, content))
            logger.progress(i + 2)

    _save_cache(files)

    return files


def split_train_valid(files: List[EncodedFile],
                      is_shuffle=True) -> (List[EncodedFile], List[EncodedFile]):
    """
    Split training and validation sets
    """
    if is_shuffle:
        np.random.shuffle(files)

    total_size = sum([len(f.codes) for f in files])
    valid = []
    valid_size = 0
    while len(files) > 0:
        if valid_size > total_size * 0.15:
            break
        valid.append(files[0])
        valid_size += len(files[0].codes)
        files.pop(0)

    train_size = sum(len(f.codes) for f in files)
    if train_size < total_size * 0.60:
        raise RuntimeError("Validation set too large")

    logger.info(train_size=train_size,
                valid_size=valid_size,
                vocab=tokenizer.VOCAB_SIZE)
    return files, valid


def main():
    files, code_to_str = load_files()
    logger.info(code_to_str)


if __name__ == "__main__":
    main()
