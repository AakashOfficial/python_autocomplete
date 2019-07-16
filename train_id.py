"""
This is still work in progress

TODO: This code is hacked together to try things fast.
TODO: Needs a complete rewrite from tokenizer level
"""

import math
from typing import List, Optional, NamedTuple

import torch
import torch.nn

from parser.load import EncodedFile, split_train_valid, load_files
from lab.experiment.pytorch import Experiment
from parser import tokenizer

# Configure the experiment
from parser.batch_builder import BatchBuilder
from parser.merge_tokens import InputProcessor

EXPERIMENT = Experiment(name="id_embeddings",
                        python_file=__file__,
                        comment="With ID embeddings",
                        check_repo_dirty=False,
                        is_log_python_file=False)

logger = EXPERIMENT.logger

# device to train on
device = torch.device("cuda:1")
cpu = torch.device("cpu")


class ModelOutput(NamedTuple):
    decoded_input_logits: torch.Tensor
    decoded_predictions: Optional[torch.Tensor]
    probabilities: torch.Tensor
    logits: torch.Tensor
    hn: torch.Tensor
    cn: torch.Tensor


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
        # TODO: Use actual sequence for training
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
        self.is_evaluate = False

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
            d.length = InputProcessor.MAX_LENGTH[i]

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
        for i, di in enumerate(decoded_inputs):
            di = di.argmax(dim=-1).detach()
            if len(di.shape) == 1:
                di = di.reshape(-1, 1)
            decoded_inputs[i] = di

        embeddings_cycle: List[torch.Tensor] = self.apply_transform(encoders, decoded_inputs)

        softmax_masks = [(decoded_inputs[i] != inputs[i]).max(dim=1, keepdim=True)[0] for i in
                         range(n_inputs)]
        softmax_masks = [m.to(torch.float32) for m in softmax_masks]
        embeddings_cycle = [embeddings_cycle[i] * softmax_masks[i] for i in range(n_inputs)]

        # concatenate all the stuff
        embeddings: torch.Tensor = torch.cat(embeddings, dim=0)
        embeddings_cycle: torch.Tensor = torch.cat(embeddings_cycle, dim=0)

        if self.is_evaluate:
            # Reversed prediction
            pe = prediction_embeddings.view(-1, embedding_size)
            decoded_prediction, _ = self.apply_transform(decoders,
                                                         [pe] * len(decoders),
                                                         2)
            for i, di in enumerate(decoded_prediction):
                di = di.argmax(dim=-1).detach()
                if len(di.shape) == 1:
                    di = di.unsqueeze(-1)
                decoded_prediction[i] = di
            embedding_prediction: List[torch.Tensor] = self.apply_transform(encoders,
                                                                            decoded_prediction)
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

            embedding_prediction: torch.Tensor = torch.cat(embedding_prediction, dim=0)
            embeddings: torch.Tensor = torch.cat((embeddings, embedding_prediction),
                                                 dim=0)
        else:
            embeddings: torch.Tensor = torch.cat((embeddings, embeddings_cycle),
                                                 dim=0)
            decoded_prediction = None

        logits = torch.matmul(prediction_embeddings, embeddings.transpose(0, 1))

        probabilities = self.softmax(logits)

        return ModelOutput(decoded_input_logits, decoded_prediction,
                           probabilities, logits, hn, cn)


class Trainer:
    """
    This will maintain states, data and train/validate the model
    """

    def __init__(self, *, files: List[EncodedFile],
                 input_processor: InputProcessor,
                 model: Model,
                 loss_func, encoder_decoder_loss_funcs, optimizer,
                 eof: int,
                 batch_size: int, seq_len: int,
                 is_train: bool,
                 h0, c0):
        # Get batches
        builder = BatchBuilder(input_processor, logger)

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
            total_loss = total_loss + enc_dec_losses[-1] * 5.

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
        files = load_files()

    # files = files[:100]

    # Transform files
    with logger.section("Transform files"):
        processor = InputProcessor(logger)
        processor.gather_files(files)
        files = processor.transform_files(files)

    with logger.section("Split training and validation"):
        # Split training and validation data
        train_files, valid_files = split_train_valid(files, is_shuffle=False)

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


def run_epoch(model,
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
    steps_per_epoch = 1000

    # Next batch to train and validation
    train_batch = 0
    valid_batch = 0

    is_interrupted = False

    # Loop through steps
    for i in logger.loop(range(1, steps_per_epoch + 1)):
        # Last batch to train and validate
        train_batch_limit = len(trainer.batches) * min(1., i / steps_per_epoch)
        valid_batch_limit = len(validator.batches) * min(1., i / steps_per_epoch)

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
                if i % (steps_per_epoch // 10) == 0:
                    logger.new_line()

                # Set global step
                logger.add_global_step()

        except KeyboardInterrupt:
            is_interrupted = True

    logger.save_progress()
    logger.save_checkpoint()
    logger.new_line()
    logger.finish_loop()

    return not is_interrupted


def create_model():
    encoding_size = 256
    id_vocab = tokenizer.get_vocab_size(tokenizer.TokenType.name)
    num_vocab = tokenizer.get_vocab_size(tokenizer.TokenType.number)

    encoder_ids = LstmEncoder(vocab_size=id_vocab + 1,
                              vocab_embedding_size=256,
                              lstm_size=256,
                              lstm_layers=3,
                              encoding_size=encoding_size)
    encoder_nums = LstmEncoder(vocab_size=num_vocab + 1,
                               vocab_embedding_size=256,
                               lstm_size=256,
                               lstm_layers=3,
                               encoding_size=encoding_size)
    token_embeddings = torch.nn.Embedding(tokenizer.VOCAB_SIZE, encoding_size)
    encoder_tokens = EmbeddingsEncoder(embedding=token_embeddings)

    decoder_ids = LstmDecoder(vocab_size=id_vocab + 1,
                              lstm_size=256,
                              lstm_layers=3,
                              encoding_size=encoding_size)
    decoder_nums = LstmDecoder(vocab_size=num_vocab + 1,
                               lstm_size=256,
                               lstm_layers=3,
                               encoding_size=encoding_size)
    decoder_tokens = EmbeddingsDecoder(embedding=token_embeddings)

    model = Model(encoder_ids=encoder_ids,
                  encoder_nums=encoder_nums,
                  encoder_tokens=encoder_tokens,
                  decoder_ids=decoder_ids,
                  decoder_nums=decoder_nums,
                  decoder_tokens=decoder_tokens,
                  encoding_size=encoding_size,
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

    EXPERIMENT.start_train(False)

    # Setup logger
    for t in ['train', 'valid']:
        logger.add_indicator(f"{t}_loss", queue_limit=500, is_histogram=True)
        logger.add_indicator(f"{t}_loss_main", queue_limit=500, is_histogram=True)
        for i in range(3):
            logger.add_indicator(f"{t}_loss_enc_dec_{i}", queue_limit=500,
                                 is_print=i != 0,
                                 is_histogram=True)

    for epoch in range(100):
        if not run_epoch(model,
                         loss_func, encoder_decoder_loss_funcs, optimizer,
                         seq_len, batch_size,
                         h0, c0):
            break


if __name__ == '__main__':
    main()
