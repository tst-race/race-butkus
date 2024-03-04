"""
This module implements `AbstractModelWrapper`, an abstract base class for
accessing large language model predictions.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple


class AbstractModelWrapper(ABC):
    """
    An abstract base class for accessing large language model predictions.

    It exposes two key components of an LLM: the ability to tokenize text and
    the ability to output a next token prediction.
    """

    NAME: str

    @abstractmethod
    def __init__(
        self,
        model_dir: str,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the model using the model given by the path `model_dir`.
        """

    @abstractmethod
    def get_token(self, index: int) -> str:
        """
        Return the token associated with the given token index.
        """

    @abstractmethod
    def tokenize(self, sentence: str) -> List[int]:
        """
        Tokenize the given sentence into a list of token indices.
        """

    @abstractmethod
    def prediction(
        self,
        sentence: str,
        state: Optional[Any] = None,
        high_to_low: bool = True,
    ) -> Tuple[List[Tuple[int, float]], Any]:
        """
        Output the prediction as a cumulative probability over possible next
        tokens.

        Args:
            sentence (`str`): The prior token(s) to use for the prediction.

            state (`Optional[Any]`, optional): The model state to use. Defaults to `None`.

            high_to_low (`bool`, optional): Whether to sort the probabilities from highest to lowest. Defaults to `True`.

        Returns:
            `Tuple[List[Tuple[int, float]], Any]`: A tuple containing two values: (1) a list of `(token, cumulative probability)` pairs, and (2) the updated state.
        """
