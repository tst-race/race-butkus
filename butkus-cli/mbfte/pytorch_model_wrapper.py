"""
This module implements `PyTorchModelWrapper`, an instantiation of
`AbstractModelWrapper` that utilizes the `pytorch` library.
"""

import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import transformers
from transformers.modeling_outputs import ModelOutput
import torch

from mbfte import _logger
from mbfte.model_wrapper import AbstractModelWrapper
from mbfte.utils import (
    bool_to_symbol,
    cumulative_probabilities,
    find_split_index,
)


class PyTorchModelWrapper(AbstractModelWrapper):
    """
    An implementation of `AbstractModelWrapper` that uses a GPT-2 model through
    `pytorch`.
    """

    NAME: str = "pytorch"

    def __init__(
        self, model_dir: str, _temperature: float = 0.8, float64: bool = True
    ) -> None:
        _logger.info("Temperature: IGNORED!")
        _logger.info("64-bit floats? %s", bool_to_symbol(float64))
        if float64:
            # This is needed for cross-platform compatibility!
            torch.set_default_dtype(torch.float64)

        config = transformers.GPT2Config(
            scale_attn_by_inverse_layer_idx=True, reorder_and_upcast_attn=True
        )

        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

        if model_dir == "PRETRAINED":
            _logger.info("Loading pretrained GPT-2 model")
            self.model = transformers.GPT2LMHeadModel(config)
            self.model = self.model.from_pretrained("gpt2")
        else:
            _logger.info("Loading model from %s", model_dir)
            self.model = torch.jit.load(os.path.join(model_dir, "gpt2.ptl"))

    def get_token(self, index: int) -> str:
        token = self.tokenizer.decode(index)
        if TYPE_CHECKING:
            assert isinstance(token, str)
        return token

    def tokenize(self, sentence: str) -> List[int]:
        indices = self.tokenizer(sentence)["input_ids"]
        if TYPE_CHECKING:
            assert isinstance(indices, List)
        return indices

    def prediction(
        self,
        sentence: str,
        state: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        high_to_low: bool = True,
    ) -> Tuple[List[Tuple[int, float]], Any]:
        inputs = self.tokenizer(sentence, return_tensors="pt")
        if state is None:
            state = _make_dummy_past()
        outputs: ModelOutput = self.model(inputs["input_ids"], past_key_values=state)
        # Note: We cannot use `outputs.logits` and `outputs.past_key_values`
        # here because those fields may be not be available for locally trained
        # models.
        logits = outputs[0].tolist()[0]
        state = outputs[1]

        predictions = _get_predictions(logits)
        return (cumulative_probabilities(predictions, high_to_low), state)


def _make_dummy_past() -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    def make_tensor_pair() -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, 12, 0, 64),  # pylint: disable=no-member
            torch.zeros(1, 12, 0, 64),  # pylint: disable=no-member
        )

    return tuple(make_tensor_pair() for _ in range(12))


def _get_predictions(
    logits: Any,
) -> List[Tuple[int, float]]:
    # logits should be a 2D array where for each token in the input (in order), we
    # have a score for each potential next token. We only care about the scoring
    # associated with the last token in the input stream.
    v = _softmax(logits[-1])
    nonzeros = np.argwhere(v)
    return [(x[0], v[x[0]]) for x in nonzeros]


def _softmax(lst: List[float]) -> List[float]:
    ex = np.exp(lst - np.max(lst))
    result = ex / ex.sum(axis=0)
    if TYPE_CHECKING:
        assert isinstance(result, list)
    return result


class TopKModelWrapper(AbstractModelWrapper):
    """
    A wrapper around `PyTorchModelWrapper` that returns only the top `k`
    predictions.
    """

    NAME: str = "top-k"

    def __init__(self, model_dir: str, k: int = 100, **kwargs: Any):
        self._torch = PyTorchModelWrapper(model_dir, **kwargs)
        self._k: int = k

    def get_token(self, index: int) -> str:
        return self._torch.get_token(index)

    def tokenize(self, sentence: str) -> List[int]:
        return self._torch.tokenize(sentence)

    def prediction(
        self, sentence: str, state: Optional[Any] = None, high_to_low: bool = True
    ) -> Tuple[List[Tuple[int, float]], Any]:
        (prediction, state) = self._torch.prediction(sentence, state, high_to_low)
        # Restrict the predictions to the top `k` values.
        total: float = prediction[self._k - 1][1]
        for i in range(self._k):
            prediction[i] = (prediction[i][0], prediction[i][1] / total)
        return (prediction[: self._k], state)


class VariableTopKModelWrapper(AbstractModelWrapper):
    """
    A wrapper around `PyTorchModelWrapper` that returns the top `k` predictions,
    where `k` is determined by a precision value.

    In more detail, given a precision value of `n` which corresponds to the
    significant digits to keep, `k` is set to the point at which the difference
    between consecutive token probabilities is less than `10^-n`.
    """

    NAME: str = "variable-top-k"

    def __init__(self, model_dir: str, precision: Optional[int] = None, **kwargs: Any):
        self._torch = PyTorchModelWrapper(model_dir, **kwargs)
        self._threshold: Optional[float] = None
        if precision is not None:
            self._threshold = 10 ** (-precision)

    def get_token(self, index: int) -> str:
        return self._torch.get_token(index)

    def tokenize(self, sentence: str) -> List[int]:
        return self._torch.tokenize(sentence)

    def prediction(
        self, sentence: str, state: Optional[Any] = None, high_to_low: bool = True
    ) -> Tuple[List[Tuple[int, float]], Any]:
        (cumprobs, state) = self._torch.prediction(sentence, state, high_to_low)
        split_index: Optional[int] = find_split_index(
            cumprobs, threshold=self._threshold, high_to_low=high_to_low
        )
        assert split_index is not None
        total: float = cumprobs[split_index][1]
        for i in range(split_index):
            cumprobs[i] = (cumprobs[i][0], cumprobs[i][1] / total)
        # Set the last probability to 1.
        cumprobs[split_index - 1] = (cumprobs[split_index - 1][0], 1.0)
        return (cumprobs[:split_index], state)
