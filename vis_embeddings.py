from pathlib import Path

import numpy as np
import torch.nn

import lab.embeddings
import parser.load
from parser.merge_tokens import InputProcessor
import train_id
from lab.experiment.pytorch import Experiment

# Configure the experiment
from parser import tokenizer

EXPERIMENT = Experiment(name="id_embeddings",
                        python_file=__file__,
                        comment="With ID embeddings",
                        check_repo_dirty=False,
                        is_log_python_file=False)

logger = EXPERIMENT.logger

# device to train on
device = torch.device("cuda:1")
cpu = torch.device("cpu")


def create_token_array(processor, token_type: int, length: int, tokens: list):
    res = np.zeros((len(tokens), length), dtype=np.uint8)

    if token_type == 0:
        res[:, 0] = tokens
        return res

    token_type -= 1
    infos = processor.infos[token_type]
    data_array = processor.arrays[token_type]

    for i, t in enumerate(tokens):
        info = infos[t]
        res[i, :info.length] = data_array[info.offset:info.offset + info.length]

    return res


def main():
    with logger.section("Loading data"):
        files = parser.load.load_files()
        files = files[100: ]
        train_files, valid_files = parser.load.split_train_valid(files, is_shuffle=False)

    with logger.section("Create model"):
        model = train_id.create_model()

    train_files = train_files[:100]
    EXPERIMENT.add_models({'base': model})

    EXPERIMENT.start_replay()

    processor = InputProcessor(logger)
    processor.gather_files(train_files)

    id_list = [i for i in range(len(processor.infos[0]))]
    labels = [info.string for info in processor.infos[0]]
    ids = create_token_array(processor, 1, 80, id_list)
    ids = torch.tensor(ids, dtype=torch.int64, device=device)
    embeddings = model.encoder_ids(ids)
    model.decoder_ids.length = InputProcessor.MAX_LENGTH[1]
    decoded, _ = model.decoder_ids(embeddings[:120])
    decoded = decoded.argmax(dim=-1)
    embeddings = embeddings.detach().cpu().numpy()

    for i in range(decoded.shape[0]):
        print(processor.infos[0][i].string)
        coding = decoded[i]
        res = ''
        for c in coding:
            if c == 0:
                break
            c += processor.offsets[1]
            res += tokenizer.DECODE[c][0]
        print(res)

    lab.embeddings.save_embeddings(
        path=Path(EXPERIMENT.lab.path / 'logs' / 'embeddings' / 'projector'),
        embeddings=embeddings,
        images=None,
        labels=labels)


if __name__ == '__main__':
    main()
