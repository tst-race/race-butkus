"""
This module exposes the `TextCover` class, an implementation of MB-FTE for large
language models.
"""

from math import ceil, floor
from os.path import isdir
import logging
from time import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from Crypto.Random import get_random_bytes
from fixedint import MutableUInt32  # pylint: disable=no-name-in-module

from mbfte import _logger
from mbfte.crypto import RandomPadding
from mbfte.model_wrapper import AbstractModelWrapper
from mbfte.pytorch_model_wrapper import VariableTopKModelWrapper
from mbfte.utils import (
    bool_to_symbol,
    find_split_index,
    is_index_valid,
    truncate,
)


class UnableToTokenizeCovertext(Exception):
    """
    Raised when the encoded covertext does not tokenize to the same tokenization
    as produced by the default tokenizer.
    """


class SentinelCheckFailed(Exception):
    """Raised when the sentinel check fails on decoding."""


class UnableToDecryptValidCiphertext(Exception):
    """Raised when decoding fails to decrypt a ciphertext successfully."""


class ExtraBitsNotValid(Exception):
    """Raised when the extra bits appended to the ciphertext are invalid."""


class TextCover:
    """
    Bundles an `AbstractModelWrapper` with a symmetric encryption scheme to
    implement model-based format transforming encryption, where the format is
    text emitted by the `AbstractModelWrapper`.

    All initialization parameters, including model information, `seed`, `key`,
    `temperature`, etc. MUST be shared between the two parties for encoding and
    decoding to work.
    """

    def __init__(
        self,
        model_dir: str,
        model_wrapper: Type[AbstractModelWrapper],
        model_params: Dict[str, str],
        seed: str,
        key: Optional[bytes] = None,
        padding: int = 3,
        precision: Optional[int] = None,
        extra_encoding_bits: int = 8,
        flip_distribution: bool = False,
    ) -> None:
        """
        Initializes `TextCover`.

        Args:
            model_dir (`str`): A directory containing the model to use, or the special string "PRETRAINED", which utilizes a default model.
            model_wrapper (`Type[AbstractModelWrapper]`): The model wrapper to use.
            seed (`str`): The seed text.
            key (`Optional[bytes]`, optional): The symmetric key to use. If `None` it will generate a key internally. Defaults to `None`.
            padding (`int`, optional): The number of bytes of padding. Defaults to 3.
            precision (`Optional[int]`, optional): The number of digits of precision. Defaults to `None`.
            extra_encoding_bits (`int`, optional): The number of extra bits to encode to avoid potential decoding errors. This is a heuristic: if you are seeing decoding errors try increasing this value. Defaults to 8.
            flip_distribution (`bool`, optional): Whether to flip the distribution on each iteration.

        Raises:
            `FileNotFoundError`: The model cannot be found.
        """
        if key is None:
            key = get_random_bytes(32)

        if not isdir(model_dir) and model_dir != "PRETRAINED":
            raise FileNotFoundError(f"Invalid model directory: {model_dir}")
        self.model: AbstractModelWrapper
        t0 = time()
        if model_wrapper is VariableTopKModelWrapper:
            self.model = VariableTopKModelWrapper(model_dir, precision, **model_params)
        else:
            self.model = model_wrapper(model_dir, **model_params)
        t1 = time()
        _logger.info("Time to load model: %.4fs", t1 - t0)
        self._seed: str = seed
        self._padding: int = padding
        self._precision: Optional[int] = precision
        self._threshold: Optional[float]
        if self._precision is not None:
            self._threshold = 10 ** (-self._precision)
        else:
            self._threshold = None
        self._extra_encoding_bits: int = extra_encoding_bits
        # Whether to flip the cumulative probability distribution at each
        # iteration.
        self._flip_cumprob: bool = flip_distribution
        self.encrypter = RandomPadding(key, bytes_of_padding=self._padding)
        # The function to finish decryption, or `None` if decryption hasn't been
        # started yet.
        self.finish_decryption: Optional[Callable[[bytes], Optional[bytes]]] = None

    def key(self) -> bytes:
        """Return the symmetric key used."""
        return self.encrypter.key()

    def seed(self) -> str:
        """Return the seed text used."""
        return self._seed

    def encode(self, plaintext: str, complete_sentence: bool = True) -> Tuple[str, int]:
        """
        Encode a plaintext.

        Args:
            plaintext (`str`): The plaintext message to encode.
            complete_sentence (`bool`): Whether to try completing the sentence when creating the covertext. This uses a simple heuristic that checks that the last character in the covertext is a '.'.

        Returns:
            `Tuple[str, int]`: The encoded covertext and the number of model tokens used.

        Raises:
            `UnableToTokenizeCovertext`: Tokenization failed.

        """
        while True:
            try:
                nonce, ciphertext = self.encrypter.encrypt(bytes(plaintext, encoding="utf-8"))
                _logger.debug("Nonce = %r", nonce)
                _logger.debug("Ciphertext = %r", ciphertext)
                # Combine the `nonce` and `ciphertext` and encode this combined ciphertext.
                ciphertext = nonce + ciphertext
                return self._encode_fixed_width(ciphertext, complete_sentence)
            except UnableToTokenizeCovertext as e:
                pass

    def decode(self, covertext: str) -> str:
        """
        Decode a covertext.

        Args:
            covertext (`str`): The covertext to decode.

        Returns:
            `str`: The decoded plaintext message.

        Raises:
            `SentinelCheckFailed`: The sentinel check failed.
            `UnableToDecryptValidCiphertext`: Decryption failed.
            `ExtraBitsNotValid`: Extra ciphertext bits are not valid.
        """
        result = self._decode_fixed_width(covertext, False)
        if TYPE_CHECKING:
            assert isinstance(result, str)
        return result

    def check(self, covertext: str) -> bool:
        """
        Check a potential covertext for a hidden message.

        Args:
            covertext (`str`): The covertext to check.

        Returns:
            `bool`: Whether the check succeeded or not.
        """
        result = self._decode_fixed_width(covertext, True)
        if TYPE_CHECKING:
            assert isinstance(result, bool)
        return result

    @staticmethod
    def _fixed_width_bitrange(low: MutableUInt32, high: MutableUInt32) -> MutableUInt32:
        """Return `high - low + 1`, or the max difference if `high - low` equals that max."""
        assert high >= low
        return high - low + 1 if high - low != MutableUInt32.maxval else MutableUInt32(MutableUInt32.maxval)  # type: ignore

    @staticmethod
    def _fixed_width_adjust(
        low: MutableUInt32,
        bitrange: MutableUInt32,
        cumulative_prob: float,
        prev_cumulative_prob: float,
    ) -> Tuple[MutableUInt32, MutableUInt32]:
        """
        Adjust the range based on the cumulative probability of some token
        alongside the cumulative probability of the prior token in the
        distribution.
        """
        _logger.debug("cumulative prob:      %s", cumulative_prob)
        _logger.debug("prev cumulative prob: %s", prev_cumulative_prob)
        assert prev_cumulative_prob <= cumulative_prob
        high = low + floor(float(bitrange) * cumulative_prob)  # type: ignore
        # This happens when `high` overflows. In this case set it to the max
        # possible value.
        if high == MutableUInt32.minval:
            high = MutableUInt32(MutableUInt32.maxval)
        low = low + floor(float(bitrange) * prev_cumulative_prob)  # type: ignore
        assert low <= high
        return (low, high)

    def _encode_fixed_width(
        self,
        ciphertext: bytes,
        complete_sentence: bool = False,
    ) -> Tuple[str, int]:
        # This uses the approaches defined in the video
        # `<https://www.youtube.com/watch?v=EqKbT3QdtOI&list=PLU4IQLU9e_OrY8oASHx0u3IXAL9TOdidm&index=14>`_
        # and the book Introduction to Data Compression, Chapter 4
        # `<http://students.aiu.edu/submissions/profiles/resources/onlineBook/E3B9W5_data%20compression%20computer%20information%20technology.pdf>`.

        ciphertext_bits = self.encrypter.ciphertextbits(ciphertext)

        # The last token chosen based on ciphertext bits. At initialization this
        # is set to the model seed text.
        last_token: str = self._seed
        # Any past state from the model.
        state: Optional[Any] = None
        # The produced covertext.
        covertext: str = ""
        # List of token indices chosen. It is used to check whether the
        # generated covertext after tokenization would result in the same list
        # of token indices.
        token_indices: List[int] = []
        # List to store which cumulative probability index the algorithm picks
        # during encoding.
        cumprob_indices: List[int] = []
        # Max number of bits to encode.
        max_bits = len(ciphertext) * 8 + self._extra_encoding_bits
        # Keeps track of the number of bits encoded so far.
        num_encoded_bits: int = 0

        # The low bitrange for arithmetic decoding.
        low: MutableUInt32 = MutableUInt32(MutableUInt32.minval)
        # The high bitrange for arithmetic decoding.
        high: MutableUInt32 = MutableUInt32(MutableUInt32.maxval)
        # The value we are encoding, which we use when selecting the next token.
        # We maintain the invariant that `low <= encoded <= high`.
        encoded: MutableUInt32 = MutableUInt32(ciphertext_bits.get(MutableUInt32.width))

        # Whether to order the cumulative probabilities from high to low or low
        # to high.
        high_to_low: bool = True

        # Used for debugging to see whether we successfully encode the ciphertext bits.
        if _logger.isEnabledFor(logging.DEBUG):
            to_be_taken_out: str = format(encoded, f"0{MutableUInt32.width}b")
            took_out: str = ""

        # The core loop of MB-FTE encoding.
        #
        # This runs arithmetic _decoding_, using the bits of the ciphertext to
        # choose tokens produced by the model. In more detail, we loop the
        # following until we are out of ciphertext bits to decode.
        #
        # 1. We first extract the cumulative probability of the next potential
        #    token from the model, using the seed text and any past state saved
        #    by the model.
        # 2. Next, we find which token corresponds to the associated ciphertext
        #    bits. That is the token that we add to the covertext.
        # 3. Finally, we "adjust" the ciphertext bits for the next iteration.
        #
        # These steps are described in more detail below.
        done: bool = False
        niters: int = 0
        while not done:
            # This invariant should always hold.
            assert low <= encoded <= high, f"{low:08x} | {encoded:08x} | {high:08x}"

            if self._flip_cumprob:
                high_to_low = not high_to_low
            t0 = time()
            # Step 1: Call the model to get a list of tokens and their
            # associated probabilities, and then covert this into a cumulative
            # probabilities list.
            cumprobs, state = self.model.prediction(last_token, state, high_to_low)
            _logger.debug("# tokens: %d", len(cumprobs))
            while len(cumprobs) == 0:
                _logger.debug("No tokens to embed into!")
                # Try again! This "should" work because even though `last_token`
                # hasn't changed, `state` has.
                #
                # It might create some bogus text though!
                cumprobs, state = self.model.prediction(last_token, state, high_to_low)
                _logger.debug("# tokens: %d", len(cumprobs))
            split_index: Optional[int] = find_split_index(
                cumprobs, self._threshold, high_to_low
            )

            # Step 2: Find the first token index with cumulative probability
            # greater than the probability scaled by the bitrange.
            i: int = 0
            bitrange = TextCover._fixed_width_bitrange(low, high)
            while (
                i < len(cumprobs) - 1
                and truncate(cumprobs[i][1], self._precision)
                < (encoded - low) / bitrange
            ):
                i += 1

            # Get the token associated with the given token index and add it to
            # `covertext`.
            last_token = self.model.get_token(cumprobs[i][0])
            covertext += last_token
            token_indices.append(cumprobs[i][0])
            cumprob_indices.append(i)

            valid: bool = is_index_valid(i, split_index, high_to_low)

            _logger.debug("High to low? %s", bool_to_symbol(high_to_low))
            _logger.debug("Index:       %d [%s]", i, split_index)
            _logger.debug("Valid?       %s", bool_to_symbol(valid))

            if valid:
                # Step 3: "Adjust" the range and find out how many ciphertext bits
                # we used to make our choice in Step 2.
                (low, high) = TextCover._fixed_width_adjust(
                    low,
                    bitrange,
                    truncate(cumprobs[i][1], self._precision),
                    truncate(cumprobs[i - 1][1] if i > 0 else 0.0, self._precision),
                )
                _logger.debug("Low:  %08x", low)
                _logger.debug("High: %08x", high)
                while True:
                    if low[-1] == high[-1]:
                        next_encoding_bit = ciphertext_bits.get(1)
                        num_encoded_bits += 1

                        low = low << 1  # type: ignore
                        high = (high << 1) | 1  # type: ignore
                        encoded = (encoded << 1) | next_encoding_bit  # type: ignore

                        if _logger.isEnabledFor(logging.DEBUG):
                            took_out += to_be_taken_out[0]
                            to_be_taken_out = to_be_taken_out[1:] + str(
                                next_encoding_bit
                            )
                    elif low[-2] == 1 and high[-2] == 0:
                        next_encoding_bit = ciphertext_bits.get(1)
                        num_encoded_bits += 1

                        low[-2] = low[-1]
                        high[-2] = high[-1]
                        encoded[-2] = encoded[-1]

                        low = low << 1  # type: ignore
                        high = high << 1 | 1  # type: ignore
                        encoded = encoded << 1 | next_encoding_bit  # type: ignore

                        if _logger.isEnabledFor(logging.DEBUG):
                            took_out += to_be_taken_out[0]
                            to_be_taken_out = to_be_taken_out[1:] + str(
                                next_encoding_bit
                            )
                    else:
                        break

                    # Check if we are done encoding everything we need to.
                    if num_encoded_bits >= max_bits:
                        if complete_sentence:
                            if covertext[-1] == ".":
                                done = True
                                break
                        else:
                            done = True
                            break

            t1 = time()
            niters += 1
            _logger.info(
                "Iteration %d: %.4fs (%d / %d)",
                niters,
                t1 - t0,
                num_encoded_bits,
                max_bits,
            )

        (nciphertextbits, nextrabits) = ciphertext_bits.stats()
        _logger.info("# ciphertext bits: %d", nciphertextbits)
        _logger.info("# extra bits: %d", nextrabits)

        _logger.debug(
            "Ciphertext bits: %s", "".join(format(byte, "08b") for byte in ciphertext)
        )

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug("Encoded bits:    %s", took_out)

        # Encoding fails if tokenizing the covertext doesn't produce the same
        # tokens produced during encoding. This is because when decoding we use
        # the default tokenizer (as is done in this check), so if we end up with
        # different tokens than what the encoder used, we're going to get the
        # wrong ciphertext out.
        if self.model.tokenize(covertext) != token_indices:
            _logger.error("Encoding failed: Unable to tokenize produced covertext")
            _logger.debug("  Covertext:·%s·", covertext)
            _logger.debug("  Tokenization: %s", self.model.tokenize(covertext))
            _logger.debug("  Expected:     %s", token_indices)
            raise UnableToTokenizeCovertext

        _logger.info("Done encoding! Number of iterations: %d", niters)
        _logger.debug("Covertext:·%s·", covertext)
        return (covertext, niters)

    def _decode_fixed_width(
        self,
        covertext: str,
        verify: bool = False,
    ) -> Union[bool, str]:
        # The low bitrange for arithmetic encoding.
        low: MutableUInt32 = MutableUInt32(MutableUInt32.minval)
        # The high bitrange for arithmetic encoding.
        high: MutableUInt32 = MutableUInt32(MutableUInt32.maxval)
        # Reset the "finish decryption" function.
        self.finish_decryption = None
        # The (eventual) ciphertext, stored as an integer.
        encoded: int = 0
        # The number of (arithmetic) encoded bits.
        total_encoded: int = 0
        # Whether arithmetic encoding has underflowed.
        underflow_counter: int = 0

        # The core loop of MB-FTE decoding.
        #
        # This runs arithmetic _encoding_, using the tokens of the covertext to
        # derive the underlying ciphertext bits used to make that token choice.
        # In more detail, we loop the following until we are out of covertext
        # tokens.
        #
        # 1. We first "adjust" the range to determine the ciphertext bits we
        #    need to extract using the covertext token's probability.
        # 2. We do the actual ciphertext bit extraction.
        for cumprob, prev_cumprob in self._cumprob_gen(covertext):
            bitrange = TextCover._fixed_width_bitrange(low, high)
            (low, high) = TextCover._fixed_width_adjust(
                low,
                bitrange,
                truncate(cumprob, self._precision),
                truncate(prev_cumprob, self._precision),
            )
            _logger.debug("Low:  %08x", low)
            _logger.debug("High: %08x", high)
            # The number of bits encoded for this token.
            nencoded: int = 0
            while True:
                if low[-1] == high[-1]:
                    value = int(low[-1])
                    encoded = (encoded << 1) + value
                    nencoded += 1

                    while underflow_counter > 0:
                        encoded = (encoded << 1) + (not value)
                        nencoded += 1
                        underflow_counter -= 1

                    low = low << 1  # type: ignore
                    high = (high << 1) | 1  # type: ignore
                elif low[-2] == 1 and high[-2] == 0:
                    low[-2] = low[-1]
                    high[-2] = high[-1]

                    low = low << 1  # type: ignore
                    high = high << 1 | 1  # type: ignore
                    underflow_counter += 1
                else:
                    if verify:
                        result = self._try_decrypt(
                            encoded, total_encoded + nencoded, verify
                        )
                        if result is not None:
                            return result
                    break
            total_encoded += nencoded
            _logger.debug("Number of bits encoded: %d / %d", nencoded, total_encoded)

        _logger.debug(
            "Encoded: %s",
            _bytes_to_bits(_convert_int_to_bytes(encoded, total_encoded)),
        )

        result = self._try_decrypt(encoded, total_encoded, verify)
        if result is not None:
            return result

        if verify:
            # If we're verifying and get here, it means we never got enough
            # ciphertext bits to even _try_ verifying. Which means verification
            # fails.
            return False

        _logger.error("Decoding failed: Unable to decrypt valid ciphertext")
        raise UnableToDecryptValidCiphertext

    def _cumprob_gen(
        self,
        sentence: str,
    ) -> Generator[Tuple[float, float], None, List[int]]:
        """
        A generator for extracting the cumulative probabilities for some input
        string.

        Args:
            sentence (`str`): The input string.

        Returns:
            `List[int]`: A list of token indices chosen, where the token index
            corresponds to the cumulative probability (that is, a token index of
            0 means the most likely token, _not_ the token corresponding to the
            first entry in the token list).

        Yields:
            `Generator[Tuple[float, float, int], None, None]`: The cumulative
            probability of the current token, the cumulative probability of
            the previous token, and whether the token is "valid".
        """
        t0 = time()
        token_indices: List[Any] = self.model.tokenize(sentence)
        t1 = time()
        _logger.info("`tokenize`: %.4fs", t1 - t0)
        # A list containing the token indices corresponding to the cumulative
        # probability.
        cumprob_token_indices: List[int] = []
        # Whether to order the cumulative probabilities from high to low or low
        # to high.
        high_to_low: bool = True

        # The last token we've extracted. Defaults to the seed for the first
        # iteration.
        last_token: str = self._seed
        # The past state of the model.
        state: Optional[Any] = None
        for i, token_index in enumerate(token_indices, start=1):
            result: Optional[Tuple[float, float]] = None

            if self._flip_cumprob:
                high_to_low = not high_to_low

            t0 = time()
            cumprobs, state = self.model.prediction(last_token, state, high_to_low)
            _logger.debug("# tokens: %d", len(cumprobs))
            while len(cumprobs) == 0:
                _logger.debug("No tokens to decode from!")
                cumprobs, state = self.model.prediction(last_token, state, high_to_low)
                _logger.debug("# tokens: %d", len(cumprobs))

            # Find the probability associated with the token index of interest.
            for j, (token_index_, probability) in enumerate(cumprobs):
                if token_index_ == token_index:
                    split_index: Optional[int] = find_split_index(
                        cumprobs, self._threshold, high_to_low
                    )

                    last_token = self.model.get_token(token_index)
                    cumprob_token_indices.append(j)
                    valid: bool = is_index_valid(j, split_index, high_to_low)

                    _logger.debug("High to low? %s", bool_to_symbol(high_to_low))
                    _logger.debug("Index:       %d [%s]", j, split_index)
                    _logger.debug("Valid?       %s", bool_to_symbol(valid))

                    if valid:
                        result = (
                            probability,
                            cumprobs[j - 1][1] if j > 0 else 0.0,
                        )
                    break
            t1 = time()
            _logger.info("Iteration %d: %.4fs", i, t1 - t0)

            if result is None:
                continue
            yield result

        return cumprob_token_indices

    def _try_decrypt(
        self, encoded: int, total_encoded: int, verify: bool = False
    ) -> Optional[Union[bool, str]]:
        """
        Try decrypting `encoded`, returning `None` on failure, and either a bool
        if `verify` is set to `True` or the decrypted string if `verify` is set
        to `False`.

        Note! It is essential that before calling this method in a loop,
        `self.finish_decryption` is set to `None`!

        Args:
            encoded (`int`): The value to decrypt, encoded as an integer.
            total_encoded (`int`): The total number of ciphertext bits in `encoded`.
            verify (`bool`, optional): Whether to only verify decryption or not. Defaults to False.

        Raises:
            `SentinelCheckFailed`: The sentinel check failed.
            `ExtraBitsNotValid`: The extra bits appended to the ciphertext are not valid.

        Returns:
            `Optional[Union[bool, str]]`: Either a bool denoting whether verification succeeded (if `verify = True`) or the plaintext (if `verify = False`), or `None` if decryption failed.
        """
        # Only try decrypting if we have enough of the ciphertext.
        if floor(total_encoded / 8) >= self.encrypter.bytes_to_check():
            ct = _convert_int_to_bytes(encoded, total_encoded)

            if not self.finish_decryption:
                self.finish_decryption = self.encrypter.begin_decryption(
                    ct[: self.encrypter.bytes_to_check()]
                )
                _logger.debug(
                    "Bits checked: %s %s",
                    _bytes_to_bits(ct[: self.encrypter.bytes_to_check()]),
                    bool_to_symbol(self.finish_decryption is not None),
                )

            if verify:
                return self.finish_decryption is not None
            else:
                if self.finish_decryption is not None:
                    # We don't know how long the properly encrypted message is.
                    # So try decrypting all possible lengths!
                    for end in range(self.encrypter.bytes_of_nonce(), len(ct) + 1):
                        plaintext = self.finish_decryption(
                            ct[self.encrypter.bytes_of_nonce() : end]
                        )
                        if plaintext is not None:
                            # Now, we need to check that the bonus bits are valid.
                            n = total_encoded - end * 8
                            n = n if n > 0 else 0
                            if self.encrypter.check_extra_bits(ct[end:], ct[:end], n):
                                return plaintext.decode("utf-8")
                            else:
                                _logger.error("Decoding failed: Extra bits not valid")
                                raise ExtraBitsNotValid
                    _logger.warning(
                        "No valid ciphertext found up to length %d bytes", len(ct)
                    )
                else:
                    _logger.error("Decoding failed: Sentinel check failed")
                    raise SentinelCheckFailed
        return None

    def cumprob_indices(self, sentence: str) -> List[int]:
        """
        Compute the cumulative probability token indices for an input string.

        Args:
            sentence (`str`): The input string.

            seed (`str`): The model seed to use.

        Returns:
           `List[int]`: The resulting token indices.
        """
        generator = self._cumprob_gen(sentence)
        try:
            while True:
                next(generator)
        except StopIteration as result:
            return result.value  # type: ignore


def _convert_int_to_bytes(ct: int, nbits: int) -> bytes:
    """
    Convert an integer encoding a ciphertext into its byte representation.
    """
    if nbits % 8 != 0:
        ct = ct << 8 - (nbits % 8)
    return ct.to_bytes(length=int(ceil(nbits / 8)), byteorder="big")


def _bytes_to_bits(b: bytes) -> str:
    """Convert a byte string to its bit representation."""
    return "".join(format(byte, "08b") for byte in b)
