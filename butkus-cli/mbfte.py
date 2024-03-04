"""
A program for running MB-FTE.
"""

from io import TextIOWrapper
import json
from statistics import mean, median, stdev
from time import time
from typing import Any, Callable, Dict, List, Optional, Type
import logging
import random
import sys

import click
import numpy as np
import torch

from mbfte.textcover import (
    ExtraBitsNotValid,
    SentinelCheckFailed,
    TextCover,
    UnableToDecryptValidCiphertext,
    UnableToTokenizeCovertext,
)
from mbfte.model_wrapper import AbstractModelWrapper
from mbfte.pytorch_model_wrapper import (
    PyTorchModelWrapper,
    TopKModelWrapper,
    VariableTopKModelWrapper,
)
from mbfte.utils import bool_to_symbol, random_string


_logger = logging.getLogger(__name__)


def _encode_covertext(covertext: str) -> str:
    # Covertexts might contain newlines. If we are writing these covertexts to a
    # newline-separated file this causes problems. So the hack is to replace the
    # newlines by some hopefully never-used character.
    #
    # Note! We are hoping here that the covertext does not contain '🤞'!!!
    return covertext.replace("\n", "🤞")


def _decode_covertext(encoded: str) -> str:
    return encoded.replace("🤞", "\n")


def roundtrip(
    plaintext: str,
    key: bytes,
    model: TextCover,
    model2: TextCover,
    complete_sentence: bool = True,
) -> Dict[str, Any]:
    """
    Test the full encode-decode cycle and collect any relevant metrics.

    Args:
        plaintext (`str`): The plaintext to encode.
        key (`bytes`): The key to use.
        model (`TextCover`): The "good" model to use.
        model2 (`TextCover`): The "bad" model to use to validate that decoding fails when using the wrong model.
        complete_sentence (`bool`, optional): Whether covertexts should be complete sentences. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary containing various metrics.
    """
    _logger.info("Encoding plaintext")
    start = time()
    (covertext, niters) = model.encode(plaintext, complete_sentence)
    end = time()
    encode_time = end - start

    _logger.info("Checking covertext")
    start = time()
    ok = model.check(covertext)
    end = time()
    good_check_time = end - start
    if not ok:
        _logger.error("Decoding check failed.")
        _logger.error("  Plaintext:·%s·", plaintext)
        _logger.error("  Covertext:·%s·", covertext)
        _logger.error("  Key:·%s·", key.hex())
        assert False, "Decoding check should never fail"

    _logger.info("Decoding covertext")
    start = time()
    plaintext_ = model.decode(covertext)
    end = time()
    decode_time = end - start

    _logger.info("Decoding covertext with wrong model")
    start = time()
    ok2 = model2.check(covertext)
    end = time()
    bad_check_time = end - start
    # Check using `model2` should fail, since the seed is different.
    if ok2:
        _logger.error("Decoding check succeeded on wrong model.")
        _logger.error("  Plaintext:·%s·", plaintext)
        _logger.error("  Covertext:·%s·", covertext)
        _logger.error("  Key:·%s·", key.hex())
        assert False, "Decoding check on wrong model should never succeed"

    if plaintext_ != plaintext:
        _logger.error("Decoding check succeeded but decoding failed.")
        _logger.error("  Plaintext:·%s·", plaintext)
        _logger.error("  Covertext:·%s·", covertext)
        _logger.error("  Decoded plaintext:·%s·", plaintext_)
        _logger.error("  Key:·%s·", key.hex())
        assert False, "Decoding should never fail"

    _logger.info("Tokenizing covertext")
    token_indices = model.model.tokenize(covertext)
    _logger.info("Computing cumulative probability indices")
    cumprob_indices = model.cumprob_indices(covertext)

    return {  # all times in seconds
        "encode time": encode_time,
        "good check time": good_check_time,
        "decode time": decode_time,
        "bad check time": bad_check_time,
        "plaintext": plaintext,
        "covertext": covertext,
        "niters": niters,
        "expansion": len(covertext) / len(plaintext),
        "key": key.hex(),
        "tokens": token_indices,
        "cumprobs": cumprob_indices,
    }


def benchmark(
    length: int,
    n_trials: int,
    save_covertexts: Optional[TextIOWrapper],
    seed: str,
    model_dir: str,
    model_wrapper: Type[AbstractModelWrapper],
    model_params: Dict[str, str],
    padding: int,
    precision: Optional[int],
    extra_encoding_bits: int,
    flip_distribution: bool,
    complete_sentence: bool,
) -> Dict[str, Any]:
    """
    Collect MB-FTE statistics.
    """

    # `random.randbytes` only available in versions >= 3.9.
    # key = random.randbytes(32)
    key = random.getrandbits(256).to_bytes(length=32, byteorder="little")

    model = TextCover(
        model_dir,
        model_wrapper,
        model_params,
        seed,
        key,
        padding,
        precision,
        extra_encoding_bits,
        flip_distribution,
    )

    model2 = TextCover(
        model_dir,
        model_wrapper,
        model_params,
        # Make the seed different, so the model is different.
        seed + " ",
        key,
        padding,
        precision,
        extra_encoding_bits,
        flip_distribution,
    )

    trials = []
    covertexts: List[str] = []
    plaintexts: List[str] = []
    tokens: List[List[int]] = []
    cumprobs: List[List[int]] = []
    failure_unable_to_tokenize_covertext: int = 0
    failure_sentinel_check_failed: int = 0
    failure_unable_to_decrypt: int = 0
    _logger.info("Key: %s", key.hex())
    for i in range(n_trials):
        _logger.info("Trial #%d", i + 1)
        plaintext = random_string(length)
        _logger.info("Plaintext:·%s·", plaintext)
        try:
            t = roundtrip(plaintext, key, model, model2, complete_sentence)
            trials.append(t)
            _logger.info("Result: %s", json.dumps(t))
            covertexts.append(t["covertext"])
            plaintexts.append(t["plaintext"])
            tokens.append(t["tokens"])
            cumprobs.append(t["cumprobs"])
        except UnableToTokenizeCovertext:
            failure_unable_to_tokenize_covertext += 1
        except SentinelCheckFailed:
            failure_sentinel_check_failed += 1
        except UnableToDecryptValidCiphertext:
            assert False, "We should always be able to decrypt a valid ciphertext."
        except ExtraBitsNotValid:
            assert False, "Extra bits should always be good."
    # Extract non-`None` data from `trials`.
    f: Callable[[str], List[float]] = lambda name: [
        float(i)  # type: ignore
        for i in filter(lambda x: x is not None, (t.get(name) for t in trials))
    ]
    encode_time = f("encode time")
    good_check_time = f("good check time")
    decode_time = f("decode time")
    bad_check_time = f("bad check time")
    expansion = f("expansion")
    niters = f("niters")
    # Collect relevant statistics.
    compute_mean = lambda values: mean(values) if len(values) > 0 else float("nan")
    compute_stdev = lambda values: stdev(values) if len(values) > 1 else float("nan")
    compute_median = lambda values: median(values) if len(values) > 0 else float("nan")
    successful = (
        n_trials
        - failure_unable_to_tokenize_covertext
        - (failure_sentinel_check_failed + failure_unable_to_decrypt)
    )
    result: Dict[str, Any] = {
        "model": {
            "type": model_wrapper.NAME,
            "dir": model_dir,
        },
        "key": key.hex(),
        "plaintext length": length,
        "trials": {
            "total": n_trials,
            "successful": successful,
            "rate": f"{int((successful / n_trials) * 100)}%",
        },
        "encode failures": {
            "unable to tokenize covertext": failure_unable_to_tokenize_covertext,
        },
        "decode failures": {
            "total": failure_sentinel_check_failed + failure_unable_to_decrypt,
            "sentinel check failed": failure_sentinel_check_failed,
            "unable to decrypt": failure_unable_to_decrypt,
        },
        "encode": {
            "mean": f"{compute_mean(encode_time):.4f} +- {compute_stdev(encode_time):.4f}",
            "median": f"{compute_median(encode_time):.4f}",
        },
        "decode": {
            "mean": f"{compute_mean(decode_time):.4f} +- {compute_stdev(decode_time):.4f}",
            "median": f"{compute_median(decode_time):.4f}",
        },
        "good check": {
            "mean": f"{compute_mean(good_check_time):.4f} +- {compute_stdev(good_check_time):.4f}",
            "median": f"{compute_median(good_check_time):.4f}",
        },
        "bad check": {
            "mean": f"{compute_mean(bad_check_time):.4f} +- {compute_stdev(bad_check_time):.4f}",
            "median": f"{compute_median(bad_check_time):.4f}",
        },
        "expansion": {
            "mean": f"{compute_mean(expansion):.4f} +- {compute_stdev(expansion):.4f}",
            "median": f"{compute_median(expansion):.4f}",
        },
        "niters": {
            "mean": f"{compute_mean(niters):.4f} +- {compute_stdev(niters):.4f}",
            "median": f"{compute_median(niters):.4f}",
        },
    }
    click.echo("Stats:")
    click.echo(json.dumps(result, indent=4))
    click.echo()
    for i, (plaintext, covertext, tokens_, cumprobs_) in enumerate(
        zip(plaintexts, covertexts, tokens, cumprobs), start=1
    ):
        click.echo(f"#{i}·{plaintext}·")
        click.echo(f"#{i}·{covertext}·")
        click.echo(f"#{i} Token indices:    {tokens_}")
        click.echo(f"#{i} Cum prob indices: {cumprobs_}")
    if save_covertexts is not None:
        # Write out useful parameters first.
        save_covertexts.write(
            f"# key={key.hex()}; "
            f"seed={seed}; "
            f"model_dir={model_dir}; "
            f"model_wrapper={model_wrapper.NAME}; "
            f"model_params={model_params}; "
            f"padding={padding}; "
            f"precision={precision}; "
            f"flip_distribution={flip_distribution}\n"
        )
        for covertext in covertexts:
            save_covertexts.write(f"{_encode_covertext(covertext)}\n")
    return result


def encode(
    key: bytes,
    plaintext: Optional[str],
    plaintext_file: Optional[TextIOWrapper],
    save_covertexts: Optional[TextIOWrapper],
    seed: str,
    model_dir: str,
    model_wrapper: Type[AbstractModelWrapper],
    model_params: Dict[str, str],
    padding: int,
    precision: Optional[int],
    extra_encoding_bits: int,
    flip_distribution: bool,
    complete_sentence: bool,
) -> List[str]:
    """Run the encode operation on a plaintext or plaintext file."""
    model = TextCover(
        model_dir,
        model_wrapper,
        model_params,
        seed,
        key,
        padding,
        precision,
        extra_encoding_bits,
        flip_distribution,
    )

    covertexts: List[str] = []

    def _encode(plaintext: str) -> None:
        (covertext, _) = model.encode(plaintext, complete_sentence)
        covertexts.append(covertext)
        click.echo(f"Produced covertext:·{covertext}·")

    if plaintext is not None:
        _encode(plaintext)
    if plaintext_file is not None:
        for i, plaintext in enumerate(plaintext_file, start=1):
            if plaintext is not None:
                # Remove the newline.
                plaintext = plaintext[:-1]
                click.echo(f"#{i}:·{plaintext}·")
                _encode(plaintext)
    if save_covertexts is not None:
        for covertext in covertexts:
            save_covertexts.write(f"{_encode_covertext(covertext)}\n")
    return covertexts


def decode(
    key: bytes,
    covertext: Optional[str],
    covertext_file: Optional[TextIOWrapper],
    verify: bool,
    seed: str,
    model_dir: str,
    model_wrapper: Type[AbstractModelWrapper],
    model_params: Dict[str, str],
    padding: int,
    precision: Optional[int],
    extra_encoding_bits: int,
    flip_distribution: bool,
) -> None:
    """Run the decode operation on a covertext or covertext file."""
    model = TextCover(
        model_dir,
        model_wrapper,
        model_params,
        seed,
        key,
        padding,
        precision,
        extra_encoding_bits,
        flip_distribution,
    )

    def _check(covertext: str) -> bool:
        ok: bool = model.check(covertext)
        if ok:
            click.echo("Verification succeeded 👍")
        else:
            click.echo("Verification failed ❌")
        return ok

    def _decode(covertext: str) -> str:
        plaintext = model.decode(covertext)
        click.echo(f"Produced plaintext:·{plaintext}·")
        return plaintext

    if verify:
        ok: bool = True
        if covertext is not None:
            ok_ = _check(covertext)
            ok = ok and ok_
        if covertext_file is not None:
            for covertext in covertext_file:
                if covertext is not None:
                    ok_ = _check(covertext)
                    ok = ok and ok_
    else:
        if covertext is not None:
            _ = _decode(covertext)
        if covertext_file is not None:
            for i, covertext in enumerate(covertext_file):
                if covertext is not None:
                    if i == 0:
                        # Skip the first entry if it starts with a `#`.
                        if covertext.startswith("#"):
                            continue
                    # Remove the newline.
                    covertext = covertext[:-1]
                    covertext = _decode_covertext(covertext)
                    click.echo(f"#{i}:·{covertext}·")
                    _ = _decode(covertext)


def encode_decode(
    key: bytes,
    plaintext: str,
    seed: str,
    model_dir: str,
    model_wrapper: Type[AbstractModelWrapper],
    model_params: Dict[str, str],
    padding: int,
    verify: bool,
    precision: Optional[int],
    extra_encoding_bits: int,
    flip_distribution: bool,
    complete_sentence: bool,
) -> None:
    """
    For testing a one-off run (for example, if decoding fails and we want to
    try to diagnose why).
    """
    covertexts = encode(
        key,
        plaintext,
        None,
        None,
        seed,
        model_dir,
        model_wrapper,
        model_params,
        padding,
        precision,
        extra_encoding_bits,
        flip_distribution,
        complete_sentence,
    )
    decode(
        key,
        covertexts[0],
        None,
        verify,
        seed,
        model_dir,
        model_wrapper,
        model_params,
        padding,
        precision,
        extra_encoding_bits,
        flip_distribution,
    )


class _LogLevel(click.ParamType):
    name = "loglevel"

    def convert(self, value, param, ctx):  # type: ignore
        try:
            return getattr(logging, value)
        except AttributeError:
            self.fail(
                "Must be one of: DEBUG, INFO, WARN, ERROR",
                param,
                ctx,
            )


class _Key(click.ParamType):
    name = "key"

    def convert(self, value, param, ctx) -> bytes:  # type: ignore
        try:
            key = bytes.fromhex(value)
            if len(key) != 32:
                self.fail("Must be hex string of length 32.", param, ctx)
            return key
        except ValueError:
            self.fail("Must be hex string of length 32.", param, ctx)


class _ModelWrapper(click.ParamType):
    name = "modelwrapper"

    MODELS: Dict[str, Type[AbstractModelWrapper]] = {
        PyTorchModelWrapper.NAME: PyTorchModelWrapper,
        TopKModelWrapper.NAME: TopKModelWrapper,
        VariableTopKModelWrapper.NAME: VariableTopKModelWrapper,
    }

    def convert(self, value: str, param, ctx) -> Type[AbstractModelWrapper]:  # type: ignore
        try:
            return _ModelWrapper.MODELS[value]
        except KeyError:
            self.fail(
                f"Must be one of: {', '.join(_ModelWrapper.MODELS.keys())}",
                param,
                ctx,
            )


class _ModelWrapperParams(click.ParamType):
    name = "model-params"

    def convert(self, value: str, param, ctx) -> Dict[Any, Any]:  # type: ignore
        try:
            params: Dict[Any, Any] = json.loads(value)
            return params
        except ValueError:
            self.fail("Must be valid json", param, ctx)


class _Precision(click.ParamType):
    name = "precision"

    def convert(self, value: str, param, ctx) -> Optional[int]:  # type: ignore
        if value.lower() == "none":
            return None
        try:
            precision: int = int(value)
            if precision > 1:
                return precision
            else:
                self.fail("Precision must be an integer greater than one", param, ctx)
        except ValueError:
            self.fail("Must be an integer greater than one, or None", param, ctx)


@click.group()
@click.option(
    "--loglevel",
    metavar="LEVEL",
    help="The log level ('DEBUG', 'INFO', 'WARN', or 'ERROR').",
    type=_LogLevel(),
    default="WARN",
    show_default=True,
)
@click.option(
    "--padding",
    metavar="N",
    help="Bytes of random padding to use when encrypting.",
    type=click.IntRange(min=0),
    default=3,
    show_default=True,
)
@click.option(
    "--precision",
    metavar="T",
    help="The number of digits of precision to use.",
    type=_Precision(),
    default=None,
    show_default=True,
)
@click.option(
    "--extra-encoding-bits",
    metavar="N",
    help="The number of extra bits to encode to avoid potential decoding errors. This is a heuristic: if you are seeing decoding errors try increasing this value.",
    default=8,
    show_default=True,
)
@click.option(
    "--flip-distribution/--do-not-flip-distribution",
    help="Whether to flip the distribution after each model prediction.",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--model-type",
    metavar="TYPE",
    help=f"The model type (one of: {', '.join(_ModelWrapper.MODELS.keys())}).",
    type=_ModelWrapper(),
    default="pytorch",
    show_default=True,
)
@click.option(
    "--model-params",
    metavar="PARAMS",
    help="Model specific parameters, encoded as json.",
    type=_ModelWrapperParams(),
    default="{}",
)
@click.option(
    "--model-dir",
    metavar="DIR",
    help='The model directory, or the special text "PRETRAINED" to use a pre-trained PyTorch model.',
    type=str,
    default="PRETRAINED",
    show_default=True,
)
@click.option(
    "--seed-text",
    metavar="TEXT",
    help="Initial seed text to use in the model (generally longer is better).",
    type=str,
    default="Here is the news of the day. ",
    show_default=True,
)
@click.option(
    "--seed-randomness",
    metavar="N",
    help="Set the randomness seed to N (to be fully deterministic you also need to set `--padding 0`).",
    type=int,
    default=None,
)
@click.option(
    "--complete-sentence/--do-not-complete-sentence",
    help="Whether to complete sentences when generating covertext.",
    default=False,
    show_default=True,
)
@click.pass_context
def cli(  # type: ignore
    ctx,
    loglevel,
    padding,
    precision,
    extra_encoding_bits,
    flip_distribution,
    model_type,
    model_params,
    model_dir,
    seed_text,
    seed_randomness,
    complete_sentence,
) -> None:
    """
    Implementation of model-based format transforming encryption (MB-FTE).

    FTE is a technique for transforming a plaintext into a ciphertext such that
    the ciphertext conforms to a particular format. In MB-FTE, that format is
    the output of a large language model.

    MODEL TYPES

    We support the following model types (as specified using the `--model-type`
    flag):

    - pytorch: The "default" model which returns the full cumulative probability
      distribution of all possible tokens.

    - top-k: The pytorch model, except the cumulative probability distribution
      returned is capped to the top `k` tokens.

    - variable-top-k: The pytorch model, except the cumulative probability
      distribution returned is capped at the top tokens such that the
      probability difference between any two tokens is greater than some
      precision value, as specified using the `--precision` flag.

    MODEL PARAMETERS

    Different model types take different parameters. These are specified as a
    JSON string as an argument to the `--model-params` flag. Here we list those
    parameters.

    - float64 (bool): Enables / disables the use of 64-bit floats in the model.
      [default: True]

      - Compatible models: pytorch, top-k, variable-top-k

    - temperature (float): The model temperature to use. This is currently
      ignored internally! [default: 0.8]

      - Compatible models: pytorch, top-k, variable-top-k

    - top-k (int): What to set the top k value to. [default: 100]

      - Compatible models: top-k

    Example: To disable float64, use the following:

        --model-params '{"float64": false}'

    OTHER NOTES

    Is your decoding failing? Try adding additional encoding bits using the
    `--extra-encoding-bits` flag. Generally more bits need to be added as the
    precision decreases.
    """
    # Set up logging.
    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    logging.getLogger("mbfte").level = loglevel
    logging.getLogger(__name__).level = loglevel
    # Set up randomness.
    if seed_randomness is None:
        seed_randomness = random.getrandbits(32)
    random.seed(seed_randomness)
    np.random.seed(seed_randomness)
    torch.manual_seed(seed_randomness)

    ctx.ensure_object(dict)
    ctx.obj["seed-randomness"] = seed_randomness
    ctx.obj["padding"] = padding
    ctx.obj["precision"] = precision
    ctx.obj["extra-encoding-bits"] = extra_encoding_bits
    ctx.obj["flip-distribution"] = flip_distribution
    ctx.obj["model-wrapper"] = model_type
    ctx.obj["model-params"] = model_params
    ctx.obj["model-dir"] = model_dir
    ctx.obj["seed-text"] = seed_text
    ctx.obj["complete-sentence"] = complete_sentence


def _log_user_settings(ctx: click.Context) -> None:
    _logger.info("Model: %s", ctx.obj["model-wrapper"].NAME)
    _logger.info("Model parameters: %s", ctx.obj["model-params"])
    _logger.info("Model directory: %s", ctx.obj["model-dir"])
    _logger.info("Randomness seed: %s", ctx.obj["seed-randomness"])
    _logger.info("Padding: %s", ctx.obj["padding"])
    _logger.info("Digits of precision: %s", ctx.obj["precision"])
    _logger.info("Number of extra encoding bits: %s", ctx.obj["extra-encoding-bits"])
    _logger.info("Seed text:·%s·", ctx.obj["seed-text"])
    _logger.info("Complete sentences? %s", bool_to_symbol(ctx.obj["complete-sentence"]))
    _logger.info("Flip distribution? %s", bool_to_symbol(ctx.obj["flip-distribution"]))


@cli.command("encode")
@click.argument("key", type=_Key())
@click.option(
    "--plaintext",
    metavar="PLAINTEXT",
    help="The plaintext to encode.",
    type=str,
    default=None,
)
@click.option(
    "--plaintext-file",
    metavar="FILE",
    help="File containing plaintexts to encode.",
    type=click.File(mode="r"),
    default=None,
)
@click.option(
    "--save-covertexts",
    metavar="FILE",
    help="Output generated covertexts to a file.",
    type=click.File(mode="w"),
    default=None,
)
@click.pass_context
def cmd_encode(
    ctx: click.Context,
    plaintext: Optional[str],
    plaintext_file: Optional[TextIOWrapper],
    save_covertexts: Optional[TextIOWrapper],
    key: bytes,
) -> None:
    """Run encode using KEY."""
    _log_user_settings(ctx)
    try:
        encode(
            key,
            plaintext,
            plaintext_file,
            save_covertexts,
            ctx.obj["seed-text"],
            ctx.obj["model-dir"],
            ctx.obj["model-wrapper"],
            ctx.obj["model-params"],
            ctx.obj["padding"],
            ctx.obj["precision"],
            ctx.obj["extra-encoding-bits"],
            ctx.obj["flip-distribution"],
            ctx.obj["complete-sentence"],
        )
    except UnableToTokenizeCovertext:
        sys.exit(1)


@cli.command("decode")
@click.argument("key", type=_Key())
@click.option(
    "--covertext",
    metavar="COVERTEXT",
    help="The covertext to decode.",
    type=str,
    default=None,
)
@click.option(
    "--covertext-file",
    metavar="FILE",
    help="File containing covertexts to decode.",
    type=click.File(mode="r"),
    default=None,
)
@click.option(
    "--verify",
    help="Only verify that a covertext can be decoded successfully.",
    is_flag=True,
    default=False,
)
@click.pass_context
def cmd_decode(
    ctx: click.Context,
    key: bytes,
    covertext: Optional[str],
    covertext_file: Optional[TextIOWrapper],
    verify: bool,
) -> None:
    """Run decode using KEY."""
    _log_user_settings(ctx)
    _logger.info("Verify only? %s", bool_to_symbol(verify))
    try:
        decode(
            key,
            covertext,
            covertext_file,
            verify,
            ctx.obj["seed-text"],
            ctx.obj["model-dir"],
            ctx.obj["model-wrapper"],
            ctx.obj["model-params"],
            ctx.obj["padding"],
            ctx.obj["precision"],
            ctx.obj["extra-encoding-bits"],
            ctx.obj["flip-distribution"],
        )
    except (SentinelCheckFailed, UnableToDecryptValidCiphertext):
        sys.exit(1)


@cli.command("encode-decode")
@click.argument("plaintext", type=str)
@click.argument("key", type=_Key())
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help="Whether to just verify a covertext decodes successfully.",
)
@click.pass_context
def cmd_encode_decode(
    ctx: click.Context, plaintext: str, key: bytes, verify: bool
) -> None:
    """Run encode and decode on PLAINTEXT and KEY."""
    _log_user_settings(ctx)
    _logger.info("Verify only? %s", bool_to_symbol(verify))
    try:
        encode_decode(
            key,
            plaintext,
            ctx.obj["seed-text"],
            ctx.obj["model-dir"],
            ctx.obj["model-wrapper"],
            ctx.obj["model-params"],
            ctx.obj["padding"],
            ctx.obj["precision"],
            ctx.obj["extra-encoding-bits"],
            ctx.obj["flip-distribution"],
            ctx.obj["complete-sentence"],
            verify,
        )
    except (
        UnableToTokenizeCovertext,
        SentinelCheckFailed,
        UnableToDecryptValidCiphertext,
    ):
        sys.exit(1)


@cli.command("benchmark")
@click.option(
    "--length",
    metavar="N",
    help="Length of each plaintext message.",
    type=int,
    default=10,
    show_default=True,
)
@click.option(
    "--ntrials",
    metavar="N",
    help="Number of trials to run.",
    type=int,
    default=10,
    show_default=True,
)
@click.option(
    "--save-covertexts",
    metavar="FILE",
    help="Output generated covertexts to a file.",
    type=click.File(mode="w"),
    default=None,
)
@click.pass_context
def cmd_benchmark(
    ctx: click.Context,
    length: int,
    ntrials: int,
    save_covertexts: Optional[TextIOWrapper],
) -> None:
    """Run benchmarks.

    This command collects statistics across several runs of encode and decode
    and outputs them to STDOUT.
    """
    _log_user_settings(ctx)
    benchmark(
        length,
        ntrials,
        save_covertexts,
        ctx.obj["seed-text"],
        ctx.obj["model-dir"],
        ctx.obj["model-wrapper"],
        ctx.obj["model-params"],
        ctx.obj["padding"],
        ctx.obj["precision"],
        ctx.obj["extra-encoding-bits"],
        ctx.obj["flip-distribution"],
        ctx.obj["complete-sentence"],
    )


if __name__ == "__main__":
    try:
        cli()  # pylint: disable=no-value-for-parameter
    except FileNotFoundError as e:
        _logger.critical(e)
        sys.exit(1)
