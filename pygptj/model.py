#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a simple Python API around [gpt-j](https://github.com/ggerganov/ggml/tree/master/examples/gpt-j/main.cpp)
"""

import logging
import time
from pathlib import Path
import random
from typing import Callable, Tuple, Union, Generator

__author__ = "abdeladim-s"
__github__ = "https://github.com/abdeladim-s/pygptj"
__copyright__ = "Copyright 2023, "
__license__ = "MIT"

import logging
import _pygptj as pp
from pygptj._logger import set_log_level
import numpy as np

class Model:
    """
    GPT-J model

    Example usage
    ```python
    from pygptj.model import Model

    model = Model(ggml_model='path/to/ggml/model')
    for token in model.generate("Tell me a joke ?"):
        print(token, end='', flush=True)
    ```
    """
    _new_text_callback = None
    _logits_callback = None

    def __init__(self,
                 model_path: str,
                 prompt_context: str = '',
                 prompt_prefix: str = '',
                 prompt_suffix: str = '',
                 log_level: int = logging.ERROR):
        """
        :param model_path: The path to a gpt-j `ggml` model
        :param prompt_context: the global context of the interaction
        :param prompt_prefix: the prompt prefix
        :param prompt_suffix: the prompt suffix
        :param log_level: logging level
        """
        # set logging level
        set_log_level(log_level)
        self._ctx = None

        if not Path(model_path).is_file():
            raise Exception(f"File {model_path} not found!")

        self.model_path = model_path

        self._model = pp.gptj_model()
        self._vocab = pp.gpt_vocab()

        # load model
        self._load_model()

        # gpt params
        self.gpt_params = pp.gptj_gpt_params()
        self.hparams = pp.gptj_hparams()

        self.res = ""

        self.logits = []

        self._n_past = 0
        self.prompt_cntext = prompt_context
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix

        self._prompt_context_tokens = []
        self._prompt_prefix_tokens = []
        self._prompt_suffix_tokens = []

        self.reset()

    def _load_model(self):
        """
        Helper function to load the model
        """
        pp.gptj_model_load(self.model_path, self._model, self._vocab)
        # fix UnicodeDecode errors
        vocab = self._load_vocab()
        self._vocab.token_to_id = vocab[0]
        self._vocab.id_to_token = vocab[1]

    def _load_vocab(self):
        """
        Reloads the vocab to fix UnicodeDecode errors
        """
        token_to_id = {}
        id_to_token = {}

        with open(self.model_path, "rb") as fin:
            skip_bytes = 4 * 8
            fin.read(skip_bytes)
            n_vocab_bytes = fin.read(4)
            n_vocab = int.from_bytes(n_vocab_bytes, byteorder='little')
            for i in range(n_vocab):
                len_bytes = fin.read(4)
                len_word = int.from_bytes(len_bytes, byteorder='little')
                word = fin.read(len_word).decode('utf-8', errors='ignore')  # ignore non unicode chars
                token_to_id[word] = i
                id_to_token[i] = word

        return token_to_id, id_to_token

    def _call_new_text_callback(self, text_bytes) -> None:
        """
        Internal new_segment_callback, it just calls the user's callback with the `Segment` object
        :return: None
        """
        if Model._new_text_callback is not None:
            try:
                text = text_bytes.decode("utf-8")
                Model._new_text_callback(text)
                self.res += text
            except UnicodeDecodeError:
                logging.warning(f"UnicodeDecodeError of bytes {text_bytes}")
        # save res

    def _call_logits_callback(self, logits: np.ndarray):
        """
        Internal logits_callback that saves the logit representation at each token.
        :return: None
        """
        self.logits.append(logits.tolist())

        if Model._logits_callback is not None:
            Model._logits_callback(logits)

    def generate(self,
                 prompt: str,
                 n_predict: Union[None, int] = None,
                 antiprompt: str = None,
                 seed: int = None,
                 n_threads: int = 4,
                 top_k: int = 40,
                 top_p: float = 0.9,
                 temp: float = 0.9,
                 ) -> Generator:
        """
         Runs GPT-J inference and yields new predicted tokens

        :param prompt: The prompt :)
        :param n_predict: if n_predict is not None, the inference will stop if it reaches `n_predict` tokens, otherwise
                          it will continue until `end of text` token
        :param antiprompt: aka the stop word, the generation will stop if this word is predicted,
                           keep it None to handle it in your own way
        :param seed: random seed
        :param n_threads: The number of CPU threads
        :param top_k: top K sampling parameter
        :param top_p: top P sampling parameter
        :param temp: temperature

        :return: Tokens generator
        """
        if seed is None or seed < 0:
            seed = int(time.time())

        logging.info(f'seed = {seed}')

        if self._n_past == 0 or antiprompt is None:
            # add the prefix to the context
            embd_inp = self._prompt_prefix_tokens + pp.gpt_tokenize(self._vocab, prompt) + self._prompt_suffix_tokens
        else:
            # do not add the prefix again as it is already in the previous generated context
            embd_inp = pp.gpt_tokenize(self._vocab, prompt) + self._prompt_suffix_tokens

        if n_predict is not None:
            n_predict = min(n_predict, self.hparams.n_ctx - len(embd_inp))
        logging.info(f'Number of tokens in prompt = {len(embd_inp)}')

        embd = []
        # add global context for the first time
        if self._n_past == 0:
            for tok in self._prompt_context_tokens:
                embd.append(tok)

        # consume input tokens
        for tok in embd_inp:
            embd.append(tok)

        # determine the required inference memory per token:
        mem_per_token = 0
        logits, mem_per_token = pp.gptj_eval(self._model, n_threads, 0, [0, 1, 2, 3], mem_per_token)

        i = len(embd) - 1
        id = 0
        if antiprompt is not None:
            sequence_queue = []
            stop_word = antiprompt.strip()

        while id != 50256:  # end of text token
            if n_predict is not None:  # break the generation if n_predict
                if i >= (len(embd_inp) + n_predict):
                    break
            i += 1
            # predict
            if len(embd) > 0:
                try:
                    logits, mem_per_token = pp.gptj_eval(self._model, n_threads, self._n_past, embd, mem_per_token)
                    self.logits.append(logits)
                except Exception as e:
                    print(f"Failed to predict\n {e}")
                    return

            self._n_past += len(embd)
            embd.clear()

            if i >= len(embd_inp):
                # sample next token
                n_vocab = self.hparams.n_vocab
                t_start_sample_us = int(round(time.time() * 1000000))
                id = pp.gpt_sample_top_k_top_p(self._vocab, logits[-n_vocab:], top_k, top_p, temp, seed)
                if id == 50256:  # end of text token
                    break
                # add the token to the context
                embd.append(id)
                token = self._vocab.id_to_token[id]
                # antiprompt
                if antiprompt is not None:
                    if token == '\n':
                        sequence_queue.append(token)
                        continue
                    if len(sequence_queue) != 0:
                        if stop_word.startswith(''.join(sequence_queue).strip()):
                            sequence_queue.append(token)
                            if ''.join(sequence_queue).strip() == stop_word:
                                break
                            else:
                                continue
                        else:
                            # consume sequence queue tokens
                            while len(sequence_queue) != 0:
                                yield sequence_queue.pop(0)
                            sequence_queue = []

                yield token

    def cpp_generate(self,
                     prompt: str,
                     new_text_callback: Callable[[str], None] = None,
                     logits_callback: Callable[[np.ndarray], None] = None,
                     n_predict: int = 128,
                     seed: int = -1,
                     n_threads: int = 4,
                     top_k: int = 40,
                     top_p: float = 0.9,
                     temp: float = 0.9,
                     n_batch: int = 8,
                     ) -> str:
        """
        Runs the inference to cpp generate function

        :param prompt: the prompt
        :param new_text_callback: a callback function called when new text is generated, default `None`
        :param logits_callback: a callback function to access the logits on every inference
        :param n_predict: number of tokens to generate
        :param seed: The random seed
        :param n_threads: Number of threads
        :param top_k: top_k sampling parameter
        :param top_p: top_p sampling parameter
        :param temp: temperature sampling parameter
        :param n_batch: batch size for prompt processing

        :return: the new generated text
        """
        self.gpt_params.prompt = prompt
        self.gpt_params.n_predict = n_predict
        self.gpt_params.seed = seed
        self.gpt_params.n_threads = n_threads
        self.gpt_params.top_k = top_k
        self.gpt_params.top_p = top_p
        self.gpt_params.temp = temp
        self.gpt_params.n_batch = n_batch

        # assign new_text_callback
        self.res = ""
        Model._new_text_callback = new_text_callback

        # assign _logits_callback used for saving logits, token by token
        Model._logits_callback = logits_callback

        # run the prediction
        pp.gptj_generate(self.gpt_params, self._model, self._vocab, self._call_new_text_callback,
                         self._call_logits_callback)
        return self.res

    def braindump(self, path: str) -> None:
        """
        Dumps the logits to .npy
        :param path: Output path
        :return: None
        """
        np.save(path, np.asarray(self.logits))

    def reset(self) -> None:
        """
        Resets the context
        :return: None
        """
        self._n_past = 0
        self._prompt_context_tokens = pp.gpt_tokenize(self._vocab, self.prompt_cntext)
        self._prompt_prefix_tokens = pp.gpt_tokenize(self._vocab, self.prompt_prefix)
        self._prompt_suffix_tokens = pp.gpt_tokenize(self._vocab, self.prompt_suffix)

    @staticmethod
    def get_params(params) -> dict:
        """
        Returns a `dict` representation of the params
        :return: params dict
        """
        res = {}
        for param in dir(params):
            if param.startswith('__'):
                continue
            res[param] = getattr(params, param)
        return res
