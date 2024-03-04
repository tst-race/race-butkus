"""
Useful utility functions.
"""

from operator import itemgetter
import random
import string
from typing import List, Optional, Tuple


def cumulative_probabilities(
    probabilities: List[Tuple[int, float]], high_to_low: bool = True
) -> List[Tuple[int, float]]:
    """
    Given a list of probabilities in `(integer, probability)` format, return the
    list sorted by cumulative probability (highest to lowest).

    Args:
        probabilities (`List[Tuple[int, float]]`): The list of probabilities.

        high_to_low (`bool`, optional): Whether to sort the probabilities from highest to lowest. Defaults to `True`.

    Returns:
        `List[Tuple[int, float]]`: The list sorted by cumulative probability.
    """
    probabilities.sort(key=itemgetter(1), reverse=high_to_low)
    cumulative_probs: List[Tuple[int, float]] = []
    for i, (index, probability) in enumerate(probabilities):
        cumprob: float
        if i == 0:
            cumprob = probability
        elif i == len(probabilities) - 1:
            cumprob = 1.0
        else:
            cumprob = cumulative_probs[i - 1][1] + probability
        cumulative_probs.append((index, cumprob))
    return cumulative_probs


def truncate(probability: float, ndigits_of_precision: Optional[int]) -> float:
    """
    Truncate the given probability to the given precision. Values below the threshold are set to zero.

    Args:
        probability (`float`): The probability to truncate.
        ndigits_of_precision (`Optional[int]`): The number of digits of precision.

    Returns:
        `float`: The truncated probability.
    """
    if ndigits_of_precision is None:
        return probability
    else:
        assert ndigits_of_precision > 0
        threshold: float = 10 ** (-ndigits_of_precision)
        if probability > threshold:
            # The `+ 2` for the `0.xxxxx` in the string representationg.
            return float(str(probability)[: ndigits_of_precision + 2])
            # You'd think the below would work, but it doesn't! The suspicion is
            # the f-formatting ends up doing some rounding behind the scenes.
            #
            # return float(f"{probability:.{ndigits_of_precision + 1}f}"[:-1])
        else:
            return 0.0


def find_split_index(
    cumprobs: List[Tuple[int, float]],
    threshold: Optional[float] = None,
    high_to_low: bool = True,
) -> Optional[int]:
    """
    Given a cumulative probabilities list, find the index such that all values
    below the index increase by more than `threshold`, and all values above
    index increase by less than `threshold`. If no such point is found, return
    `None`.
    """
    if threshold is None:
        return None
    else:
        for i in range(len(cumprobs)):  # pylint: disable=consider-using-enumerate
            probability: float = cumprobs[i][1] - (cumprobs[i - 1][1] if i > 0 else 0.0)
            if (high_to_low and probability < threshold) or (
                not high_to_low and probability >= threshold
            ):
                return i
        return len(cumprobs)


def is_index_valid(index: int, split_index: Optional[int], high_to_low: bool) -> bool:
    """
    Returns `True` if `index` falls within the appropriate range, given the
    split point and ordering of the cumulative probability distribution.
    """
    return (
        split_index is None
        or (high_to_low and index < split_index)
        or (not high_to_low and index >= split_index)
    )


def bool_to_symbol(b: bool) -> str:
    """Output a bool as a nice unicode symbol."""
    #return "✔" if b else "✘"
    return "ok" if b else "not ok"


def random_string(num: int) -> str:
    """Return a random string with `num` characters."""
    return "".join(
        random.choice(string.ascii_letters + string.digits + " ") for _ in range(num)
    )
