import gc
import math
from typing import List

import numpy as np
import torch
import torch.nn

import parser.load
from lab.experiment.pytorch import Experiment
from simple_model import SimpleLstmModel
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


def prepare_batch(x, batch_size, seq_len):
    """
    Prepare flat data into batches to be ready for the model to consume
    """
    x = np.reshape(x, (batch_size, seq_len))
    x = np.transpose(x, (1, 0))

    return x


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


class Trainer:
    """
    This will maintain states, data and train/validate the model
    """

    def __init__(self, *, files: List[parser.load.EncodedFile],
                 model, loss_func, optimizer,
                 eof: int,
                 batch_size: int, seq_len: int,
                 is_train: bool,
                 h0, c0):
        # Get batches
        x, y = get_batches(files, eof,
                           batch_size=batch_size,
                           seq_len=seq_len)
        del files

        # Covert data to PyTorch tensors
        self.x = x
        self.y = y

        # Initial state
        self.hn = h0
        self.cn = c0

        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.p = None
        self.is_train = is_train

    def run(self, i):
        # Get model output
        x = torch.tensor(self.x[i], device=device, dtype=torch.int64)
        y = torch.tensor(self.y[i], device=device, dtype=torch.int64)
        self.p, logits, (self.hn, self.cn) = self.model(x, self.hn, self.cn)

        # Flatten outputs
        logits = logits.view(-1, self.p.shape[-1])
        yi = y.view(-1)

        # Calculate loss
        loss = self.loss_func(logits, yi)

        # Store the states
        self.hn = self.hn.detach()
        self.cn = self.cn.detach()

        if self.is_train:
            # Take a training step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            logger.store("train_loss", loss.cpu().data.item())
        else:
            logger.store("valid_loss", loss.cpu().data.item())


def get_trainer_validator(model, loss_func, optimizer, seq_len, batch_size, h0, c0):
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
                            optimizer=optimizer,
                            is_train=False,
                            seq_len=seq_len,
                            batch_size=batch_size,
                            h0=h0,
                            c0=c0,
                            eof=0)

    del valid_files

    return trainer, validator, batches


def run_epoch(epoch, model, loss_func, optimizer, seq_len, batch_size, h0, c0):
    trainer, validator, batches = get_trainer_validator(model, loss_func, optimizer,
                                                        seq_len, batch_size,
                                                        h0, c0)

    gc.collect()

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
    lstm_size = 1024
    lstm_layers = 3
    batch_size = 32
    seq_len = 32
    is_half = False

    with logger.section("Create model"):
        # Create model
        model = SimpleLstmModel(encoding_size=tokenizer.VOCAB_SIZE,
                                embedding_size=tokenizer.VOCAB_SIZE,
                                lstm_size=lstm_size,
                                lstm_layers=lstm_layers)

        # Use half precision
        if is_half:
            model.half()

        # Move model to `device`
        model.to(device)

        # Create loss function and optimizer
        loss_func = torch.nn.CrossEntropyLoss()
        if is_half:
            optimizer = torch.optim.Adam(model.parameters(), eps=1e-5)
        else:
            optimizer = torch.optim.Adam(model.parameters())

    # Initial state is 0
    if is_half:
        dtype = torch.float16
    else:
        dtype = torch.float32
    h0 = torch.zeros((lstm_layers, batch_size, lstm_size), device=device, dtype=dtype)
    c0 = torch.zeros((lstm_layers, batch_size, lstm_size), device=device, dtype=dtype)

    # Specify the model in [lab](https://github.com/vpj/lab) for saving and loading
    EXPERIMENT.add_models({'base': model})

    # Start training scratch (step '0')
    EXPERIMENT.start_train(True)

    # Setup logger indicators
    logger.add_indicator("train_loss", queue_limit=500, is_histogram=True)
    logger.add_indicator("valid_loss", queue_limit=500, is_histogram=True)

    for epoch in range(100):
        if not run_epoch(epoch, model, loss_func, optimizer,
                         seq_len, batch_size,
                         h0, c0):
            break


if __name__ == '__main__':
    main_train()
