"""
This is still work in progress

TODO: This code is hacked together to try things fast.
TODO: Needs refactoring and cleaning up.
"""

import math
from typing import List, Optional, NamedTuple, Dict

import numpy as np
import torch
import torch.nn

import parser.load
from parser.load import EncodedFile
from lab.experiment.pytorch import Experiment
from parser import tokenizer

# Configure the experiment

EXPERIMENT = Experiment(name="id_embeddings",
                        python_file=__file__,
                        comment="With ID embeddings",
                        check_repo_dirty=False,
                        is_log_python_file=False)

logger = EXPERIMENT.logger

# device to train on
device = torch.device("cuda:1")
cpu = torch.device("cpu")

TYPE_MASK_BASE = 1 << 20


class Batch(NamedTuple):
    x: np.ndarray
    y: Optional[np.ndarray]
    x_type: np.ndarray
    y_type: Optional[np.ndarray]
    y_idx: Optional[np.ndarray]
    tokens: np.ndarray
    ids: np.ndarray
    nums: np.ndarray


class ModelOutput(NamedTuple):
    decoded_input_logits: torch.Tensor
    probabilities: torch.Tensor
    logits: torch.Tensor
    hn: torch.Tensor
    cn: torch.Tensor


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


class LstmEncoder(torch.nn.Module):
    def __init__(self, *,
                 vocab_size,
                 vocab_embedding_size,
                 lstm_size,
                 lstm_layers,
                 encoding_size):
        super().__init__()

        self.h0 = torch.nn.Parameter(torch.zeros((lstm_layers, 1, lstm_size)))
        self.c0 = torch.nn.Parameter(torch.zeros((lstm_layers, 1, lstm_size)))

        self.embedding = torch.nn.Embedding(vocab_size, vocab_embedding_size)
        self.lstm = torch.nn.LSTM(input_size=vocab_embedding_size,
                                  hidden_size=lstm_size,
                                  num_layers=lstm_layers)
        self.output_fc = torch.nn.Linear(2 * lstm_size * lstm_layers, encoding_size)

    def forward(self, x: torch.Tensor):
        # shape of x is [seq, batch, feat]
        if len(x.shape) == 2:
            batch_size, seq_len = x.shape
            x = x.transpose(0, 1)
            x = self.embedding(x)
        else:
            batch_size, seq_len, _ = x.shape
            x = x.transpose(0, 1)

            weights = self.embedding.weight
            x = torch.matmul(x, weights)
            # x = x.unsqueeze(-1)
            # while weights.dim() < x.dim():
            #     weights = weights.unsqueeze(0)
            # x = x * weights
            # x = torch.sum(x, dim=-2)

        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()

        _, (hn, cn) = self.lstm(x, (h0, c0))
        state = torch.cat((hn, cn), dim=2)
        state.transpose_(0, 1)
        state = state.reshape(batch_size, -1)
        encoding = self.output_fc(state)

        return encoding


class LstmDecoder(torch.nn.Module):
    def __init__(self, *,
                 vocab_size,
                 lstm_size,
                 lstm_layers,
                 encoding_size):
        super().__init__()

        self.input_fc = torch.nn.Linear(encoding_size, 2 * lstm_size * lstm_layers)
        self.lstm = torch.nn.LSTM(input_size=vocab_size,
                                  hidden_size=lstm_size,
                                  num_layers=lstm_layers)
        self.output_fc = torch.nn.Linear(lstm_size, vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.length = 0

    @property
    def device(self):
        return self.output_fc.weight.device

    def forward(self, encoding: torch.Tensor):
        # shape of x is [seq, batch, feat]
        batch_size, encoding_size = encoding.shape
        encoding = self.input_fc(encoding)
        encoding = encoding.reshape(batch_size, self.lstm.num_layers, 2 * self.lstm.hidden_size)
        encoding.transpose_(0, 1)
        h0 = encoding[:, :, :self.lstm.hidden_size]
        c0 = encoding[:, :, self.lstm.hidden_size:]
        x = torch.zeros((1, batch_size, self.lstm.input_size), device=self.device)
        x[:, :, 0] = 1.
        h0 = h0.contiguous()
        c0 = c0.contiguous()

        decoded = []
        decoded_logits = []
        for i in range(self.length):
            out, (h0, c0) = self.lstm(x, (h0, c0))
            logits: torch.Tensor = self.output_fc(out)
            decoded_logits.append(logits.squeeze(0))
            probs = self.softmax(logits)
            decoded.append(probs.squeeze(0))
            x = probs

        decoded = torch.stack(decoded, dim=0)
        decoded.transpose_(0, 1)
        decoded_logits = torch.stack(decoded_logits, dim=0)
        decoded_logits.transpose_(0, 1)

        return decoded, decoded_logits


class EmbeddingsEncoder(torch.nn.Module):
    def __init__(self, *,
                 embedding: torch.nn.Embedding):
        super().__init__()

        self.embedding = embedding

    def forward(self, x: torch.Tensor):
        if x.shape[1] == 1:
            return self.embedding(x.view(-1))
        else:
            weights = self.embedding.weight
            return torch.matmul(x, weights)
            # x = x.unsqueeze(-1)
            # while weights.dim() < x.dim():
            #     weights = weights.unsqueeze(0)
            # value = x * weights
            # value = torch.sum(value, dim=-2)
            #
            # return value


class EmbeddingsDecoder(torch.nn.Module):
    def __init__(self, *,
                 embedding: torch.nn.Embedding):
        super().__init__()

        self.embedding = embedding
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        weights = self.embedding.weight

        logits = torch.matmul(x, weights.transpose(0, 1))
        # x = x.unsqueeze(-2)
        # while weights.dim() < x.dim():
        #     weights = weights.unsqueeze(0)
        #
        # logits = x * weights
        # logits = torch.sum(logits, dim=-1)

        return self.softmax(logits), logits


MAX_LENGTH = [1, 80, 25]


class Model(torch.nn.Module):
    def __init__(self, *,
                 encoder_ids: LstmEncoder,
                 encoder_nums: LstmEncoder,
                 encoder_tokens: EmbeddingsEncoder,
                 decoder_ids: LstmDecoder,
                 decoder_nums: LstmDecoder,
                 decoder_tokens: EmbeddingsDecoder,
                 encoding_size: int,
                 lstm_size: int,
                 lstm_layers: int):
        super().__init__()
        self.encoder_ids = encoder_ids
        self.encoder_nums = encoder_nums
        self.encoder_tokens = encoder_tokens
        self.decoder_ids = decoder_ids
        self.decoder_nums = decoder_nums
        self.decoder_tokens = decoder_tokens

        self.lstm = torch.nn.LSTM(input_size=encoding_size,
                                  hidden_size=lstm_size,
                                  num_layers=lstm_layers)
        self.output_fc = torch.nn.Linear(lstm_size, encoding_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    @staticmethod
    def apply_transform(funcs, values, n_outputs=1):
        if n_outputs == 1:
            return [funcs[i](values[i]) for i in range(len(values))]
        else:
            res = [[None for _ in range(len(values))] for _ in range(n_outputs)]
            for i in range(len(values)):
                out = funcs[i](values[i])
                assert len(out) == n_outputs
                for j in range(n_outputs):
                    res[j][i] = out[j]

            return res

    @property
    def device(self):
        return self.output_fc.weight.device

    def init_state(self, batch_size):
        h0 = torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size),
                         device=self.device)
        c0 = torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size),
                         device=self.device)

        return h0, c0

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                x_type: torch.Tensor,
                y_type: torch.Tensor,
                tokens: torch.Tensor,
                ids: torch.Tensor,
                nums: torch.Tensor,
                h0: torch.Tensor,
                c0: torch.Tensor):
        encoders = [self.encoder_tokens, self.encoder_ids, self.encoder_nums]
        decoders = [self.decoder_tokens, self.decoder_ids, self.decoder_nums]
        for i, d in enumerate(decoders):
            d.length = MAX_LENGTH[i]

        inputs = [tokens, ids, nums]
        n_inputs = len(inputs)
        embeddings: List[torch.Tensor] = self.apply_transform(encoders, inputs)

        n_embeddings, embedding_size = embeddings[0].shape
        seq_len, batch_size = x.shape
        x = x.reshape(-1)
        x_type = x_type.reshape(-1)

        x_embeddings = torch.zeros((batch_size * seq_len, embedding_size), device=self.device)
        for i in range(len(embeddings)):
            type_mask = x_type == i
            type_mask = type_mask.to(dtype=torch.int64)
            emb = embeddings[i].index_select(dim=0, index=x * type_mask)
            x_embeddings += type_mask.view(-1, 1).to(dtype=torch.float32) * emb

        x_embeddings = x_embeddings.reshape((seq_len, batch_size, embedding_size))

        out, (hn, cn) = self.lstm(x_embeddings, (h0, c0))
        prediction_embeddings = self.output_fc(out)

        # Reversed inputs
        decoded_inputs, decoded_input_logits = self.apply_transform(decoders, embeddings, 2)
        embeddings_cycle: List[torch.Tensor] = self.apply_transform(encoders, decoded_inputs)

        # softmax_masks = [(decoded_inputs[i] != inputs[i]).max(dim=1, keepdim=True) for i in
        #                  range(n_inputs)]
        # embeddings_cycle = [embeddings_cycle[i] * softmax_masks[i] for i in range(n_inputs)]

        # Reversed prediction
        # decoded_prediction, _ = self.apply_transform(decoders, prediction_embeddings, 2)
        # embedding_prediction: List[torch.Tensor] = self.apply_transform(encoders,
        #                                                                 decoded_prediction)
        # if y is not None:
        #     for i in range(n_inputs):
        #         embedding_prediction[j] *= (y_type == i)
        #     # TODO zero out if decoded_prediction is same as inputs[y]
        #     for i in range(batch_size):
        #         t: int = y_type[i]
        #         n: int = y[i]
        #         for j in range(n_inputs):
        #             if j != t:
        #                 embedding_prediction[j][i] *= 0.
        #         if inputs[t][n] == decoded_prediction[t][i]:
        #             embedding_prediction[t][i] *= 0.

        # concatenate all the stuff
        embeddings: torch.Tensor = torch.cat(embeddings, dim=0)
        embeddings_cycle: torch.Tensor = torch.cat(embeddings_cycle, dim=0)
        # embedding_prediction: torch.Tensor = torch.cat(embedding_prediction, dim=0)

        # embeddings: torch.Tensor = torch.cat((embeddings, embeddings_cycle, embedding_prediction),
        #                                      dim=0)
        embeddings: torch.Tensor = torch.cat((embeddings, embeddings_cycle),
                                             dim=0)

        logits = torch.matmul(prediction_embeddings, embeddings.transpose(0, 1))

        probabilities = self.softmax(logits)

        return ModelOutput(decoded_input_logits, probabilities, logits, hn, cn)


class InputProcessor:
    """
    TODO: We should do this at tokenizer level
    """

    def __init__(self):
        self.infos: List[List[IdentifierInfo]] = [[], []]
        self.dictionaries: List[Dict[str, int]] = [{} for _ in self.infos]
        self.arrays: List[np.ndarray] = [np.array([], dtype=np.uint8) for _ in self.infos]
        self.counts: List[int] = [0 for _ in self.infos]

    def add_to(self, type_idx: int, key: str, arr: np.ndarray):
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
                        self.add_to(type_idx, strings[type_idx],
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
                self.add_to(type_idx, strings[type_idx],
                            np.array(arrays[type_idx], dtype=np.uint8))

    def gather_files(self, files: List[parser.load.EncodedFile]):
        with logger.section("Counting", total_steps=len(files)):
            for i, f in enumerate(files):
                self.gather(f.codes)
                logger.progress(i + 1)

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
        codes = type_mask * TYPE_MASK_BASE + codes

        return codes

    def transform_files(self, files: List[parser.load.EncodedFile]) -> List[EncodedFile]:
        transformed = []
        with logger.section("Transforming", total_steps=len(files)):
            for i, f in enumerate(files):
                transformed.append(EncodedFile(f.path, self.transform(f.codes)))
                logger.progress(i + 1)

        return transformed


class BatchBuilder:
    def __init__(self, input_processor: InputProcessor):
        self.infos = input_processor.infos
        self.token_data_arrays = input_processor.arrays
        self.freqs = [self.get_frequencies(info) for info in self.infos]

    @staticmethod
    def get_frequencies(info: List[IdentifierInfo]):
        freqs = [(i.code, i.count) for i in info]
        freqs.sort(reverse=True, key=lambda x: x[1])
        return [f[0] for f in freqs]

    @staticmethod
    def get_batches(files: List[parser.load.EncodedFile],
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

        for i, f in enumerate(files):
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
        for i in range(batches):
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
        infos = self.infos[token_type]
        data_array = self.token_data_arrays[token_type]

        for i, t in enumerate(tokens):
            info = infos[t]
            res[i, :info.length] = data_array[info.offset:info.offset + info.length]

        return res

    def _get_token_sets(self, x_source: np.ndarray, y_source: np.ndarray):
        seq_len, batch_size = x_source.shape

        sets: List[set] = [set() for _ in range(len(self.infos) + 1)]

        for s in range(seq_len):
            for b in range(batch_size):
                type_idx = x_source[s, b] // TYPE_MASK_BASE
                c = x_source[s, b] % TYPE_MASK_BASE
                sets[type_idx].add(c)

                type_idx = y_source[s, b] // TYPE_MASK_BASE
                c = y_source[s, b] % TYPE_MASK_BASE
                sets[type_idx].add(c)

        for i in range(1, len(self.infos) + 1):
            sets[i] = sets[i].union(self.freqs[i - 1][:128])

        return sets

    def build_batch(self, x_source: np.ndarray, y_source: np.ndarray):
        seq_len, batch_size = x_source.shape

        lists = [list(s) for s in self._get_token_sets(x_source, y_source)]
        lists[0] = [i for i in range(tokenizer.VOCAB_SIZE)]

        dicts = [{c: i for i, c in enumerate(s)} for s in lists]

        token_data = []
        for i, length in enumerate(MAX_LENGTH):
            token_data.append(self.create_token_array(i, length, lists[i]))

        x = np.zeros_like(x_source, dtype=np.int32)
        x_type = np.zeros_like(x_source, dtype=np.int8)
        y = np.zeros_like(y_source, dtype=np.int32)
        y_type = np.zeros_like(y_source, dtype=np.int8)
        y_idx = np.zeros_like(y_source, dtype=np.int32)

        offset = np.cumsum([0] + [len(s) for s in lists])

        for s in range(seq_len):
            for b in range(batch_size):
                type_idx = x_source[s, b] // TYPE_MASK_BASE
                c = x_source[s, b] % TYPE_MASK_BASE
                x[s, b] = dicts[type_idx][c]
                x_type[s, b] = type_idx

                type_idx = y_source[s, b] // TYPE_MASK_BASE
                c = y_source[s, b] % TYPE_MASK_BASE
                y[s, b] = dicts[type_idx][c]
                y_type[s, b] = type_idx
                y_idx[s, b] = offset[type_idx] + dicts[type_idx][c]

        return Batch(x, y, x_type, y_type, y_idx,
                     token_data[0],
                     token_data[1],
                     token_data[2])

        # return [len(s) for s in sets]

    def build_batches(self, x: List[np.ndarray], y: List[np.ndarray]):
        n_batches = len(x)

        batches: List[Batch] = []
        for b in range(n_batches):
            batches.append(self.build_batch(x[b], y[b]))

        return batches


class Trainer:
    """
    This will maintain states, data and train/validate the model
    """

    def __init__(self, *, files: List[parser.load.EncodedFile],
                 input_processor: InputProcessor,
                 model: Model,
                 loss_func, encoder_decoder_loss_funcs, optimizer,
                 eof: int,
                 batch_size: int, seq_len: int,
                 is_train: bool,
                 h0, c0):
        # Get batches
        builder = BatchBuilder(input_processor)

        x, y = builder.get_batches(files, eof,
                                   batch_size=batch_size,
                                   seq_len=seq_len)

        del files

        self.batches = builder.build_batches(x, y)
        del builder

        # Initial state
        self.hn = h0
        self.cn = c0

        self.model = model
        self.loss_func = loss_func
        self.encoder_decoder_loss_funcs = encoder_decoder_loss_funcs
        self.optimizer = optimizer
        self.is_train = is_train

    def run(self, batch_idx):
        # Get model output
        batch = self.batches[batch_idx]
        x = torch.tensor(batch.x, device=device, dtype=torch.int64)
        x_type = torch.tensor(batch.x_type, device=device, dtype=torch.int64)
        if self.is_train:
            y = torch.tensor(batch.y, device=device, dtype=torch.int64)
            y_type = torch.tensor(batch.y_type, device=device, dtype=torch.int64)
        else:
            y = None
            y_type = None
        y_idx = torch.tensor(batch.y_idx, device=device, dtype=torch.int64)
        tokens = torch.tensor(batch.tokens, device=device, dtype=torch.int64)
        ids = torch.tensor(batch.ids, device=device, dtype=torch.int64)
        nums = torch.tensor(batch.nums, device=device, dtype=torch.int64)

        out: ModelOutput = self.model(x, y,
                                      x_type, y_type,
                                      tokens, ids, nums,
                                      self.hn, self.cn)

        # Flatten outputs
        logits = out.logits
        logits = logits.view(-1, logits.shape[-1])
        y_idx = y_idx.view(-1)

        # Calculate loss
        loss = self.loss_func(logits, y_idx)
        total_loss = loss
        enc_dec_losses = []

        for lf, logits, actual in zip(self.encoder_decoder_loss_funcs,
                                      out.decoded_input_logits,
                                      [tokens, ids, nums]):
            logits = logits.contiguous()
            logits = logits.view(-1, logits.shape[-1])
            yi = actual.view(-1)
            enc_dec_losses.append(lf(logits, yi))
            # TODO total_loss = total_loss +
            # So that loss and total loss aren't equal
            total_loss += enc_dec_losses[-1]

        # Store the states
        self.hn = out.hn.detach()
        self.cn = out.cn.detach()

        if self.is_train:
            # Take a training step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            loss_prefix = "train"
        else:
            loss_prefix = "valid"

        logger.store(f"{loss_prefix}_loss", total_loss.cpu().data.item())
        logger.store(f"{loss_prefix}_loss_main", loss.cpu().data.item())
        for i in range(len(enc_dec_losses)):
            logger.store(f"{loss_prefix}_loss_enc_dec_{i}", enc_dec_losses[i].cpu().data.item())


def get_trainer_validator(model, loss_func, encoder_decoder_loss_funcs,
                          optimizer, seq_len, batch_size, h0, c0):
    with logger.section("Loading data"):
        # Load all python files
        files = parser.load.load_files()

    files = files[:100]

    # Transform files
    processor = InputProcessor()
    processor.gather_files(files)
    files = processor.transform_files(files)

    with logger.section("Split training and validation"):
        # Split training and validation data
        train_files, valid_files = parser.load.split_train_valid(files, is_shuffle=False)

    # Number of batches per epoch
    batches = math.ceil(sum([len(f[1]) + 1 for f in train_files]) / (batch_size * seq_len))

    # Create trainer
    with logger.section("Create trainer"):
        trainer = Trainer(files=train_files,
                          input_processor=processor,
                          model=model,
                          loss_func=loss_func,
                          encoder_decoder_loss_funcs=encoder_decoder_loss_funcs,
                          optimizer=optimizer,
                          batch_size=batch_size,
                          seq_len=seq_len,
                          is_train=True,
                          h0=h0,
                          c0=c0,
                          eof=0)

    del train_files

    # Create validator
    with logger.section("Create validator"):
        validator = Trainer(files=valid_files,
                            input_processor=processor,
                            model=model,
                            loss_func=loss_func,
                            encoder_decoder_loss_funcs=encoder_decoder_loss_funcs,
                            optimizer=optimizer,
                            is_train=False,
                            seq_len=seq_len,
                            batch_size=batch_size,
                            h0=h0,
                            c0=c0,
                            eof=0)

    del valid_files

    return trainer, validator, batches


def run_epoch(epoch, model,
              loss_func, encoder_decoder_loss_funcs, optimizer,
              seq_len, batch_size,
              h0, c0):
    trainer, validator, batches = get_trainer_validator(model,
                                                        loss_func,
                                                        encoder_decoder_loss_funcs,
                                                        optimizer,
                                                        seq_len, batch_size,
                                                        h0, c0)

    # Number of steps per epoch. We train and validate on each step.
    steps_per_epoch = 20000

    # Next batch to train and validation
    train_batch = 0
    valid_batch = 0

    # Loop through steps
    for i in logger.loop(range(1, steps_per_epoch)):
        # Set global step
        global_step = epoch * batches + min(batches, (batches * i) // steps_per_epoch)
        logger.set_global_step(global_step)

        # Last batch to train and validate
        train_batch_limit = len(trainer.batches) * min(1., (i + 1) / steps_per_epoch)
        valid_batch_limit = len(validator.batches) * min(1., (i + 1) / steps_per_epoch)

        try:
            with logger.delayed_keyboard_interrupt():

                with logger.section("train", total_steps=len(trainer.batches), is_partial=True):
                    model.train()
                    # Train
                    while train_batch < train_batch_limit:
                        trainer.run(train_batch)
                        logger.progress(train_batch + 1)
                        train_batch += 1

                with logger.section("valid", total_steps=len(validator.batches), is_partial=True):
                    model.eval()
                    # Validate
                    while valid_batch < valid_batch_limit:
                        validator.run(valid_batch)
                        logger.progress(valid_batch + 1)
                        valid_batch += 1

                # Output results
                logger.write()

                # 10 lines of logs per epoch
                if (i + 1) % (steps_per_epoch // 10) == 0:
                    logger.new_line()

        except KeyboardInterrupt:
            # TODO Progress save doesn't work
            logger.save_progress()
            logger.save_checkpoint()
            logger.new_line()
            logger.finish_loop()
            return False

    logger.finish_loop()
    return True


def create_model():
    id_vocab = tokenizer.get_vocab_size(tokenizer.TokenType.name)
    num_vocab = tokenizer.get_vocab_size(tokenizer.TokenType.number)

    encoder_ids = LstmEncoder(vocab_size=id_vocab + 1,
                              vocab_embedding_size=256,
                              lstm_size=256,
                              lstm_layers=3,
                              encoding_size=1024)
    encoder_nums = LstmEncoder(vocab_size=num_vocab + 1,
                               vocab_embedding_size=256,
                               lstm_size=256,
                               lstm_layers=3,
                               encoding_size=1024)
    token_embeddings = torch.nn.Embedding(tokenizer.VOCAB_SIZE, 1024)
    encoder_tokens = EmbeddingsEncoder(embedding=token_embeddings)

    decoder_ids = LstmDecoder(vocab_size=id_vocab + 1,
                              lstm_size=256,
                              lstm_layers=3,
                              encoding_size=1024)
    decoder_nums = LstmDecoder(vocab_size=num_vocab + 1,
                               lstm_size=256,
                               lstm_layers=3,
                               encoding_size=1024)
    decoder_tokens = EmbeddingsDecoder(embedding=token_embeddings)

    model = Model(encoder_ids=encoder_ids,
                  encoder_nums=encoder_nums,
                  encoder_tokens=encoder_tokens,
                  decoder_ids=decoder_ids,
                  decoder_nums=decoder_nums,
                  decoder_tokens=decoder_tokens,
                  encoding_size=1024,
                  lstm_size=1024,
                  lstm_layers=3)

    # Move model to `device`
    model.to(device)

    return model


def main():
    batch_size = 32
    seq_len = 64

    with logger.section("Create model"):
        # Create model
        model = create_model()

        # Create loss function and optimizer
        loss_func = torch.nn.CrossEntropyLoss()
        encoder_decoder_loss_funcs = [torch.nn.CrossEntropyLoss() for _ in range(3)]
        optimizer = torch.optim.Adam(model.parameters())

    # Initial state is 0
    h0, c0 = model.init_state(batch_size)

    # Specify the model in [lab](https://github.com/vpj/lab) for saving and loading
    EXPERIMENT.add_models({'base': model})

    # Start training scratch
    EXPERIMENT.start_train(False)

    # Setup logger
    for t in ['train', 'valid']:
        logger.add_indicator(f"{t}_loss", queue_limit=500, is_histogram=True)
        logger.add_indicator(f"{t}_loss_main", queue_limit=500, is_histogram=True)
        for i in range(3):
            logger.add_indicator(f"{t}_loss_enc_dec_{i}", queue_limit=500, is_histogram=True)

    for epoch in range(100):
        if not run_epoch(epoch, model,
                         loss_func, encoder_decoder_loss_funcs, optimizer,
                         seq_len, batch_size,
                         h0, c0):
            break


if __name__ == '__main__':
    main()
