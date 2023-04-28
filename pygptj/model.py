#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a simple Python API around [gpt-j](https://github.com/ggerganov/ggml/tree/master/examples/gpt-j/main.cpp)
"""

import logging
from pathlib import Path
from typing import Callable

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

    def new_text_callback(text):
        print(text, end="")

    model = Model('./models/ggml-gpt4all-j.bin')
    model.generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback)
    ```
    """
    _new_text_callback = None
    _logits_callback = None

    def __init__(self,
                 model_path: str,
                 log_level: int = logging.INFO):
        """
        :param model_path: The path to a gpt-j `ggml` model
        :param log_level: logging level, set to INFO by default
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

        self.res = ""

        self.logits = []

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
            
    def braindump(self, path: str):
        np.save(path, np.asarray(self.logits))

    def generate(self,
                 prompt: str,
                 new_text_callback: Callable[[str], None] = None,
                 logits_callback: Callable = None,
                 n_predict: int = 128,
                 seed: int = -1,
                 n_threads: int = 4,
                 top_k: int = 40,
                 top_p: float = 0.9,
                 temp: float = 0.9,
                 n_batch: int = 8,
                 ) -> str:
        """
        Runs the inference to generate new text content from the prompt provided as input

        :param prompt: the prompt
        :param new_text_callback: a callback function called when new text is generated, default `None`
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
        pp.gptj_generate(self.gpt_params, self._model, self._vocab, self._call_new_text_callback, self._call_logits_callback)
        return self.res

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

