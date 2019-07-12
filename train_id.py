import gc
import math
from typing import List, Optional, NamedTuple

import numpy as np
import torch
import torch.nn

import parser.load
from lab.experiment.pytorch import Experiment
from parser import tokenizer

# Configure the experiment

EXPERIMENT = Experiment(name="simple_lstm_1000",
                        python_file=__file__,
                        comment="Simple LSTM All Data",
                        check_repo_dirty=False,
                        is_log_python_file=False)

logger = EXPERIMENT.logger

# device to train on
device = torch.device("cuda:1")
cpu = torch.device("cpu")


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

    def forward(self, x):
        # shape of x is [seq, batch, feat]
        batch_size, seq_len = x.shape[0]

        x = self.embedding(x)

        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()

        _, (hn, cn) = self.lstm(x, (h0, c0))
        state = torch.cat((hn, cn), dim=2)
        state.transpose_(0, 1)
        state = state.reshape(batch_size, -1)
        encoding = self.fc(state)

        return encoding


class LstmDecoder(torch.nn.Module):
    def __init__(self, *,
                 vocab_size,
                 lstm_size,
                 lstm_layers,
                 encoding_size):
        super().__init__()

        self.input_fx = torch.nn.Linear(encoding_size, 2 * lstm_size * lstm_layers)
        self.lstm = torch.nn.LSTM(input_size=vocab_size,
                                  hidden_size=lstm_size,
                                  num_layers=lstm_layers)
        self.output_fc = torch.nn.Linear(lstm_size, vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.length = 0

    def forward(self, encoding: torch.Tensor):
        # shape of x is [seq, batch, feat]
        batch_size, encoding_size = encoding.shape
        encoding = encoding.reshape(batch_size, self.lstm.num_layers, 2 * self.lstm.hidden_size)
        encoding.transpose_(0, 1)
        h0 = encoding[:, :, :self.lstm.hidden_size]
        c0 = encoding[:, :, self.lstm.hidden_size]
        x = torch.zeros((batch_size, self.lstm.input_size), device=self.device)
        x[:, 0] = 1.

        decoded = []
        decoded_logits = []
        for i in range(self.length):
            out, (h0, c0) = self.lstm(x, (h0, c0))
            logits = self.output_fc(out)
            decoded_logits.append(logits)
            probs = self.softmax(logits)
            decoded.append(probs)
            x = probs

        decoded = torch.stack(decoded, dim=0)
        decoded_logits = torch.stack(decoded_logits, dim=0)

        return decoded, decoded_logits


class EmbeddingsEncoder(torch.nn.Module):
    def __init__(self, *,
                 embedding: torch.nn.Embedding):
        super().__init__()

        self.embedding = embedding

    def forward(self, x):
        return self.embedding(x)


class EmbeddingsDecoder(torch.nn.Module):
    def __init__(self, *,
                 embedding: torch.nn.Embedding):
        super().__init__()

        self.embedding = embedding
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-2)
        weights = self.embedding.weight
        while weights.dim() < x.dim():
            weights = weights.unsqueeze(0)

        logits = x * weights
        logits = torch.sum(logits, dim=-1)

        return self.softmax(logits), logits


MAX_LENGTH = [80, 25, 1]


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
        self.encode_ids = encoder_ids
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
                for j in range(n_outputs):
                    res[j][i] = out

            return res

    def init_state(self, batch_size):
        h0 = torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size),
                         device=self.device)
        c0 = torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size),
                         device=self.device)

        return h0, c0

    def forward(self,
                x: torch.Tensor,
                type_mask: torch.Tensor,
                ids: torch.Tensor,
                nums: torch.Tensor,
                tokens: torch.Tensor,
                h0: torch.Tensor,
                c0: torch.Tensor,
                y: Optional[torch.Tensor]):
        encoders = [self.encode_ids, self.encode_nums, self.encode_tokens]
        decoders = [self.decode_ids, self.decode_nums, self.decode_tokens]
        for i, d in enumerate(decoders):
            d.length = MAX_LENGTH[i]

        inputs = [ids, nums, tokens]
        n_inputs = len(inputs)
        embeddings: List[torch.Tensor] = self.apply_transform(encoders, inputs)

        n_embeddings, embedding_size = embeddings[0].shape
        batch_size, seq_len = x.shape
        x = x.reshape(-1)
        type_mask = type_mask.reshape(-1, len(embeddings))

        x_embeddings = torch.zeros((batch_size * seq_len, embedding_size), device=self.device)
        for i in range(len(embeddings)):
            x_embeddings += type_mask[:, i] * embeddings[i].index_select(dim=0, index=x)

        x_embeddings = x_embeddings.reshape((batch_size, seq_len, embedding_size))

        out, (hn, cn) = self.lstm(x_embeddings, (h0, c0))
        prediction_embeddings = self.output_fc(out)

        # Reversed inputs
        decoded_inputs, decoded_input_logits = self.apply_transform(decoders, embeddings, 2)
        embeddings_cycle: List[torch.Tensor] = self.apply_transform(encoders, decoded_inputs)
        softmax_masks = [(decoded_inputs[i] != inputs[i]).max(dim=1, keepdim=True) for i in
                         range(n_inputs)]
        embeddings_cycle = [embeddings_cycle[i] * softmax_masks[i] for i in range(n_inputs)]

        # Reversed prediction
        decoded_prediction, _ = self.apply_transform(decoders, prediction_embeddings, 2)
        embedding_prediction: List[torch.Tensor] = self.apply_transform(encoders,
                                                                        decoded_prediction)
        if y is not None:
            for i in range(batch_size):
                t = y[i] // n_embeddings
                n = y[i] % n_embeddings
                for j in range(n_inputs):
                    if j != t:
                        embedding_prediction[j][i] *= 0.
                if inputs[t][n] == decoded_prediction[t][i]:
                    embedding_prediction[t][i] *= 0.

        # concatenate all the stuff
        embeddings: torch.Tensor = torch.cat(embeddings, dim=0)
        embeddings_cycle: torch.Tensor = torch.cat(embeddings_cycle, dim=0)
        embedding_prediction: torch.Tensor = torch.cat(embedding_prediction, dim=0)

        embeddings: torch.Tensor = torch.cat((embeddings, embeddings_cycle, embedding_prediction),
                                             dim=0)

        prediction_embeddings = prediction_embeddings.unsqueeze(-2)
        while embeddings.dim() < prediction_embeddings.dim():
            embeddings = embeddings.unsqueeze(0)

        logits = embeddings * prediction_embeddings
        logits = torch.sum(logits, dim=-1)

        probabilities = self.softmax(logits)

        return ModelOutput(decoded_input_logits, probabilities, logits, hn, cn)


def get_batches(files: List[parser.load.EncodedFile], eof: int, batch_size=32, seq_len=32):
    """
    Covert raw encoded files into trainin/validation batches
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

    eof = np.array([eof], dtype=np.uint8)

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
        x_batch = np.zeros((batch_size, seq_len), dtype=np.uint8)
        y_batch = np.zeros((batch_size, seq_len), dtype=np.uint8)
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


class Batch(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    type_mask: np.ndarray
    ids: np.ndarray
    nums: np.ndarray
    tokens: np.ndarray


class ModelOutput(NamedTuple):
    decoded_input_logits: torch.Tensor
    probabilities: torch.Tensor
    logits: torch.Tensor
    hn: torch.Tensor
    cn: torch.Tensor


class Trainer:
    """
    This will maintain states, data and train/validate the model
    """

    def __init__(self, *, files: List[parser.load.EncodedFile],
                 model: Model,
                 loss_func, encoder_decoder_loss_funcs, optimizer,
                 eof: int,
                 batch_size: int, seq_len: int,
                 is_train: bool,
                 h0, c0):
        # Get batches
        self.batches: List[Batch] = get_batches(files, eof,
                                                batch_size=batch_size,
                                                seq_len=seq_len)
        del files

        # Initial state
        self.hn = h0
        self.cn = c0

        self.model = model
        self.loss_func = loss_func
        self.encoder_decoder_loss_funcs = encoder_decoder_loss_funcs
        self.optimizer = optimizer
        self.is_train = is_train

    def run(self, i):
        # Get model output
        batch = self.batches[i]
        x = torch.tensor(batch.x, device=device, dtype=torch.int64)
        y = torch.tensor(batch.y, device=device, dtype=torch.int64)
        type_mask = torch.tensor(batch.type_mask, device=device, dtype=torch.int64)
        ids = torch.tensor(batch.ids, device=device, dtype=torch.int64)
        nums = torch.tensor(batch.nums, device=device, dtype=torch.int64)
        tokens = torch.tensor(batch.tokens, device=device, dtype=torch.int64)

        if self.is_train:
            model_y = y
        else:
            model_y = None

        out: ModelOutput = self.model(x, type_mask,
                                      ids, nums, tokens,
                                      self.hn, self.cn,
                                      model_y)

        # Flatten outputs
        logits = out.logits
        logits = logits.view(-1, logits.shape[-1])
        yi = y.view(-1)

        # Calculate loss
        loss = self.loss_func(logits, yi)
        total_loss = loss
        enc_dec_losses = []

        for lf, logits, actual in zip(self.encoder_decoder_loss_funcs,
                                      out.decoded_input_logits,
                                      [ids, nums, tokens]):
            logits = logits.view(-1, logits.shape[-1])
            yi = actual.view(-1)
            enc_dec_losses.append(lf(logits, yi))
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

    with logger.section("Split training and validation"):
        # Split training and validation data
        train_files, valid_files = parser.load.split_train_valid(files, is_shuffle=False)

    # Number of batches per epoch
    batches = math.ceil(sum([len(f[1]) + 1 for f in train_files]) / (batch_size * seq_len))

    # Create trainer
    with logger.section("Create trainer"):
        trainer = Trainer(files=train_files,
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
        train_batch_limit = len(trainer.x) * min(1., (i + 1) / steps_per_epoch)
        valid_batch_limit = len(validator.x) * min(1., (i + 1) / steps_per_epoch)

        try:
            with logger.delayed_keyboard_interrupt():

                with logger.section("train", total_steps=len(trainer.x), is_partial=True):
                    model.train()
                    # Train
                    while train_batch < train_batch_limit:
                        trainer.run(train_batch)
                        logger.progress(train_batch + 1)
                        train_batch += 1

                with logger.section("valid", total_steps=len(validator.x), is_partial=True):
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
            logger.save_progress()
            logger.save_checkpoint()
            logger.new_line()
            logger.finish_loop()
            return False

    logger.finish_loop()
    return True


def main_train():
    batch_size = 32
    seq_len = 32

    with logger.section("Create model"):
        # Create model
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

        # Create loss function and optimizer
        loss_func = torch.nn.CrossEntropyLoss()
        encoder_decoder_loss_funcs = [torch.nn.CrossEntropyLoss() for _ in range(3)]
        optimizer = torch.optim.Adam(model.parameters())

    # Initial state is 0
    h0, c0 = model.init_state(batch_size)

    # Specify the model in [lab](https://github.com/vpj/lab) for saving and loading
    EXPERIMENT.add_models({'base': model})

    # Start training scratch (step '0')
    EXPERIMENT.start_train(True)

    # Setup logger indicators
    logger.add_indicator("train_loss", queue_limit=500, is_histogram=True)
    logger.add_indicator("valid_loss", queue_limit=500, is_histogram=True)

    for epoch in range(100):
        if not run_epoch(epoch, model,
                         loss_func, encoder_decoder_loss_funcs, optimizer,
                         seq_len, batch_size,
                         h0, c0):
            break


if __name__ == '__main__':
    main_train()
