import time
import tokenize
from io import BytesIO
from typing import NamedTuple, List, Tuple, Optional

import numpy as np
import torch
import torch.nn

from parser.batch_builder import Batch, BatchBuilder
from parser.load import load_files, EncodedFile, split_train_valid
from parser.merge_tokens import InputProcessor
import train_id
from lab import colors
from lab.experiment.pytorch import Experiment
from parser import tokenizer

# Experiment configuration to load checkpoints
EXPERIMENT = Experiment(name="id_embeddings",
                        python_file=__file__,
                        comment="With ID embeddings",
                        check_repo_dirty=False,
                        is_log_python_file=False)

logger = EXPERIMENT.logger

# device to evaluate on
device = torch.device("cuda:1")

# Beam search
BEAM_SIZE = 100024


class Suggestions(NamedTuple):
    codes: List[List[int]]
    matched: List[int]
    scores: List[float]


class ScoredItem(NamedTuple):
    score: float
    idx: Tuple


class Predictor:
    """
    Predicts the next few characters
    """

    NEW_LINE_TOKENS = {tokenize.NEWLINE, tokenize.NL}

    def __init__(self, model):
        self.__model = model

        # Initial state
        self.h0, self.c0 = model.init_state(1)

        # Last line of source code read
        self._last_line = ""

        self._tokens: List[tokenize.TokenInfo] = []

        # Last token, because we need to input that to the model for inference
        self._last_token = 0

        # Last bit of the input string
        self._untokenized = ""

        # For timing
        self.time_add = 0
        self.time_predict = 0
        self.time_check = 0

        self.processor = InputProcessor(logger)
        self.builder = BatchBuilder(self.processor, logger)

    def __clear_tokens(self, lines: int):
        """
        Clears old lines from tokens
        """
        for i, t in enumerate(self._tokens):
            if t.type in self.NEW_LINE_TOKENS:
                lines -= 1
                if lines == 0:
                    self._tokens = self._tokens[i + 1:]
                    return

        raise RuntimeError()

    def __clear_untokenized(self, tokens):
        """
        Remove tokens not properly tokenized;
         i.e. the last token, unless it's a new line
        """

        limit = 0
        for i in reversed(range(len(tokens))):
            if tokens[i].type in self.NEW_LINE_TOKENS:
                limit = i + 1
                break
            else:
                limit = i
                break

        return tokens[:limit]

    @staticmethod
    def __get_tokens(it):
        tokens: List[tokenize.TokenInfo] = []

        try:
            for t in it:
                if t.type in tokenizer.SKIP_TOKENS:
                    continue
                if t.type == tokenize.NEWLINE and t.string == '':
                    continue
                if t.type == tokenize.DEDENT:
                    continue
                if t.type == tokenize.ERRORTOKEN:
                    continue
                tokens.append(t)
        except tokenize.TokenError as e:
            if not e.args[0].startswith('EOF in'):
                print(e)
        except IndentationError as e:
            print(e)

        return tokens

    def add(self, content):
        """
        Add a string of code, this shouldn't have multiple lines
        """
        start_time = time.time()
        self._last_line += content

        # Remove old lines
        lines = self._last_line.split("\n")
        if len(lines) > 1:
            assert len(lines) <= 3
            if lines[-1] == '':
                if len(lines) > 2:
                    self.__clear_tokens(len(lines) - 2)
                    lines = lines[-2:]
            else:
                self.__clear_tokens(len(lines) - 1)
                lines = lines[-1:]

        line = '\n'.join(lines)

        self._last_line = line

        # Parse the last line
        tokens_it = tokenize.tokenize(BytesIO(self._last_line.encode('utf-8')).readline)
        tokens = self.__get_tokens(tokens_it)

        # Remove last token
        tokens = self.__clear_untokenized(tokens)

        # Check if previous tokens is a prefix
        assert len(tokens) >= len(self._tokens)

        for t1, t2 in zip(self._tokens, tokens):
            assert t1.type == t2.type
            assert t1.string == t2.string

        # Get the untokenized string
        if len(tokens) > 0:
            assert tokens[-1].end[0] == 1
            self._untokenized = line[tokens[-1].end[1]:]
        else:
            self._untokenized = line

        # Update previous tokens and the model state
        if len(tokens) > len(self._tokens):
            self.__update_state(tokens[len(self._tokens):])
            self._tokens = tokens

        self.time_add += time.time() - start_time

    def get_predictions(self, codes_batch: List[List[int]]):
        # Sequence length and batch size
        seq_len = len(codes_batch[0])
        batch_size = len(codes_batch)

        for codes in codes_batch:
            assert seq_len == len(codes)

        # Input to the model
        x = np.array(codes_batch, dtype=np.int32)
        x = np.transpose(x, (1, 0))

        batch = self.builder.build_infer_batch(x)

        x = torch.tensor(batch.x, device=device, dtype=torch.int64)
        x_type = torch.tensor(batch.x_type, device=device, dtype=torch.int64)
        tokens = torch.tensor(batch.tokens, device=device, dtype=torch.int64)
        ids = torch.tensor(batch.ids, device=device, dtype=torch.int64)
        nums = torch.tensor(batch.nums, device=device, dtype=torch.int64)

        # Expand state
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()

        # Get predictions
        out: train_id.ModelOutput = self.__model(x, None,
                                                 x_type, None,
                                                 tokens, ids, nums,
                                                 h0, c0)

        assert out.probabilities.shape[:-1] == (seq_len, len(codes_batch))

        # Final prediction
        prediction = out.probabilities[-1, :, :]
        decoded_prediction = [d.detach().cpu().numpy() for d in out.decoded_predictions]

        return prediction.detach().cpu().numpy(), decoded_prediction

    def get_string(self, code, prev_code,
                   decoded_prediction) -> Tuple[Optional[str], Optional[int]]:
        prev_special = False
        code_special = False

        if prev_code < tokenizer.VOCAB_SIZE:
            if tokenizer.DESERIALIZE[prev_code].type == tokenizer.TokenType.keyword:
                prev_special = True
        else:
            prev_special = True
        if code < tokenizer.VOCAB_SIZE:
            if tokenizer.DESERIALIZE[code].type == tokenizer.TokenType.keyword:
                code_special = True
        else:
            code_special = True

        res = None
        new_code = None
        if code < tokenizer.VOCAB_SIZE:
            token = tokenizer.DESERIALIZE[code]
            if token.type in tokenizer.LINE_BREAK:
                return None, None
            res = tokenizer.DECODE[code][0]
            new_code = code
        else:
            code -= tokenizer.VOCAB_SIZE

            for i in range(len(self.processor.infos)):
                if code < len(self.processor.infos[i]):
                    res = self.processor.infos[i][code].string
                    new_code = (i + 1) * InputProcessor.TYPE_MASK_BASE + code
                    break
                else:
                    code -= len(self.processor.infos[i])

        if res is None:
            return None, None
            # TODO: generate unknown ids
            # Need to add these back to input_processor for the beam search
            # for idx, dp in enumerate(decoded_prediction):
            #     if code < len(dp):
            #         coding = dp[code]
            #         res = ''
            #         for c in coding:
            #             if c == 0:
            #                 break
            #             c += self.processor.offsets[idx]
            #             res += tokenizer.DECODE[c][0]
            #         break
            #     else:
            #         code -= len(dp)

        assert res is not None

        if prev_special and code_special:
            return ' ' + res, new_code
        else:
            return res, new_code

    def get_string_masked(self, code, prev_code) -> str:
        prev_special = False
        code_special = False
        prev_type_idx = prev_code // InputProcessor.TYPE_MASK_BASE
        prev_code = prev_code % InputProcessor.TYPE_MASK_BASE
        type_idx = code // InputProcessor.TYPE_MASK_BASE
        code = code % InputProcessor.TYPE_MASK_BASE

        if prev_type_idx == 0:
            if tokenizer.DESERIALIZE[prev_code].type == tokenizer.TokenType.keyword:
                prev_special = True
        else:
            prev_special = True
        if type_idx == 0:
            if tokenizer.DESERIALIZE[code].type == tokenizer.TokenType.keyword:
                code_special = True
        else:
            code_special = True

        if type_idx == 0:
            res = tokenizer.DECODE[code][0]
        else:
            res = self.processor.infos[type_idx - 1][code].string

        if prev_special and code_special:
            return ' ' + res
        else:
            return res

    def get_suggestion(self) -> str:
        # Start of with the last token
        suggestions = [Suggestions([[self._last_token]],
                                   [0],
                                   [1.])]

        # Do a beam search, up to the untokenized string length and 10 more
        for step in range(2):
            sugg = suggestions[step]
            batch_size = len(sugg.codes)

            # Break if empty
            if batch_size == 0:
                break

            # Get predictions
            start_time = time.time()
            predictions, decoded_prediction = self.get_predictions(sugg.codes)
            self.time_predict += time.time() - start_time

            start_time = time.time()
            # Get all choices
            choices = []
            for idx in range(batch_size):
                for code in range(predictions.shape[1]):
                    string, _ = self.get_string(code, sugg.codes[idx][-1], decoded_prediction)
                    if string is None:
                        continue
                    score = sugg.scores[idx] * predictions[idx, code]
                    choices.append(ScoredItem(
                        score,  # * math.sqrt(sugg.matched[idx] + len(string)),
                        (idx, code)))
            # Sort them
            choices.sort(key=lambda x: x.score, reverse=True)

            # Collect the ones that match untokenized string
            codes = []
            matches = []
            scores = []
            len_untokenized = len(self._untokenized)

            for choice in choices:
                prev_idx = choice.idx[0]
                code = choice.idx[1]

                string, new_code = self.get_string(code, sugg.codes[prev_idx][-1], decoded_prediction)
                if string is None:
                    continue

                # Previously mached length
                matched = sugg.matched[prev_idx]

                if matched >= len_untokenized:
                    # Increment the length if already matched
                    matched += len(string)
                else:
                    # Otherwise check if the new token string matches
                    unmatched = string
                    to_match = self._untokenized[matched:]

                    if len(unmatched) < len(to_match):
                        if not to_match.startswith(unmatched):
                            continue
                        else:
                            matched += len(unmatched)
                    else:
                        if not unmatched.startswith(to_match):
                            continue
                        else:
                            matched += len(unmatched)

                # Collect new item
                codes.append(sugg.codes[prev_idx] + [new_code])
                matches.append(matched)
                score = sugg.scores[prev_idx] * predictions[prev_idx, code]
                scores.append(score)

                # Stop at `BEAM_SIZE`
                if len(scores) == BEAM_SIZE:
                    break

            suggestions.append(Suggestions(codes, matches, scores))

            self.time_check += time.time() - start_time

        # Collect suggestions of all lengths
        choices = []
        for s_idx, sugg in enumerate(suggestions):
            batch_size = len(sugg.codes)
            for idx in range(batch_size):
                length = sugg.matched[idx] - len(self._untokenized)
                if length <= 1:
                    continue
                choice = sugg.scores[idx]  # * math.sqrt(length)
                choices.append(ScoredItem(choice, (s_idx, idx)))
        choices.sort(key=lambda x: x.score, reverse=True)

        # Return the best option
        for choice in choices:
            codes = suggestions[choice.idx[0]].codes[choice.idx[1]]
            res = ""
            prev = self._last_token
            for code in codes[1:]:
                string = self.get_string_masked(code, prev)
                res += string
                prev = code

            res = res[len(self._untokenized):]

            # Skip if blank
            if res.strip() == "":
                continue

            return res

        # Return blank if there are no options
        return ''

    def __update_state(self, in_tokens):
        """
        Update model state
        """
        data = tokenizer.parse(in_tokens)
        data = np.array(tokenizer.encode(data))
        self.processor.gather(data)
        data = self.processor.transform(data)

        x_source = np.concatenate(([self._last_token], data[:-1]), axis=0)
        self._last_token = data[-1]

        assert len(x_source) > 0

        x_source = np.array([x_source], dtype=np.int32)
        x_source = np.transpose(x_source, (1, 0))

        batch = self.builder.build_infer_batch(x_source)

        x = torch.tensor(batch.x, device=device, dtype=torch.int64)
        x_type = torch.tensor(batch.x_type, device=device, dtype=torch.int64)
        tokens = torch.tensor(batch.tokens, device=device, dtype=torch.int64)
        ids = torch.tensor(batch.ids, device=device, dtype=torch.int64)
        nums = torch.tensor(batch.nums, device=device, dtype=torch.int64)

        out: train_id.ModelOutput = self.__model(x, None,
                                                 x_type, None,
                                                 tokens, ids, nums,
                                                 self.h0, self.c0)

        self.h0 = out.hn.detach()
        self.c0 = out.cn.detach()


class Evaluator:
    def __init__(self, model, file: EncodedFile,
                 sample: EncodedFile,
                 skip_spaces=False):
        self.__content = self.get_content(file.codes)
        self.__skip_spaces = skip_spaces
        self.__predictor = Predictor(model)
        self.__predictor.processor.gather(sample.codes)

    @staticmethod
    def get_content(codes: np.ndarray):
        tokens = tokenizer.decode(codes)
        content = tokenizer.to_string(tokens)
        return content.split('\n')

    def eval(self):
        keys_saved = 0

        logger.info(total_keys=sum([len(c) for c in self.__content]),
                    total_lines=len(self.__content))

        for line, content in enumerate(self.__content):
            # Keep reference to rest of the line
            rest_of_line = content

            # Build the line for logging with colors
            # The line number
            logs = [(f"{line: 4d}: ", colors.BrightColor.cyan)]

            # Type the line character by character
            while rest_of_line != '':
                suggestion = self.__predictor.get_suggestion()

                # If suggestion matches
                if suggestion != '' and rest_of_line.startswith(suggestion):
                    # Log
                    logs.append((suggestion[0], colors.BrightColor.green))
                    logs.append((suggestion[1:], colors.BrightBackground.black))

                    keys_saved += len(suggestion) - 1

                    # Skip the prediction text
                    rest_of_line = rest_of_line[len(suggestion):]

                    # Add text to the predictor
                    self.__predictor.add(suggestion)

                # If the suggestion doesn't match
                else:
                    # Debug
                    end = 0
                    for i in range(min(len(rest_of_line), len(suggestion))):
                        if rest_of_line[i] != suggestion[i]:
                            end = i
                            break
                    if end > 0:
                        new_logs = logs + [(suggestion[:end], colors.Background.green),
                                           (suggestion[end:], colors.Background.red),
                                           ("#", None)]
                    else:
                        new_logs = logs + [(suggestion[end:], colors.Background.red),
                                           ("#", None)]

                    logger.log_color(new_logs)

                    # Add the next character
                    self.__predictor.add(rest_of_line[0])
                    logs.append((rest_of_line[0], None))
                    rest_of_line = rest_of_line[1:]

            # Add a new line
            self.__predictor.add("\n")

            # Log the line
            logger.log_color(logs)

        # Log time taken for the file
        logger.info(add=self.__predictor.time_add,
                    check=self.__predictor.time_check,
                    predict=self.__predictor.time_predict)
        return keys_saved


def main():
    with logger.section("Loading data"):
        files = load_files()
        train_files, valid_files = split_train_valid(files, is_shuffle=False)

    with logger.section("Create model"):
        model = train_id.create_model()
        model.is_evaluate = True

    EXPERIMENT.add_models({'base': model})

    EXPERIMENT.start_replay()

    # For debugging with a specific piece of source code
    predictor = Predictor(model)
    predictor.processor.gather(train_files[0].codes)

    for s in ['import numpy as np\n', "import "]:
        predictor.add(s)
    s = predictor.get_suggestion()

    # Evaluate all the files in validation set
    for file in valid_files[0:]:
        logger.log(str(file.path), color=colors.BrightColor.orange)
        evaluator = Evaluator(model, file, sample=valid_files[0],
                              skip_spaces=True)
        keys_saved = evaluator.eval()

        logger.info(keys_saved=keys_saved)


if __name__ == '__main__':
    main()
