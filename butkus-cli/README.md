# Model-based Format-transforming Encryption (MB-FTE)

An implementation of model-based format-transforming encryption as introduced in
[this paper](https://arxiv.org/abs/2110.07009). This implementation provides a
library and command-line interface for using MB-FTE to encode / decode messages.

Note: This implementation differs significantly from the algorithm presented in
the aforementioned paper! In particular, there are two key differences:

- We use a deterministic symmetric key scheme as opposed to the nonce-based
  symmetric key scheme used in the paper. This it to avoid needing to assume
  reliable message delivery (so that the receiver knows the correct nonce to use
  when decrypting). See `mbfte/crypto.py` for details.
- We provide a "fixed width" arithmetic encoding algorithm based on the
  algorithm presented
  [here](https://www.youtube.com/watch?v=EqKbT3QdtOI&list=PLU4IQLU9e_OrY8oASHx0u3IXAL9TOdidm&index=14).
  See `mbfte/textcover.py` for details.

## Setup

We strongly encourage using a virtual environment to isolate dependencies. It
should work with Python 3.8 or newer.

To use **for the first time**, create a new venv named "venv":

    > python -m venv venv

Then activate it:

    > source venv/bin/activate

Then install the packages that we depend on:

    (venv) > pip install -r deps.txt

Use the `deactivate` command to exit the venv.

To **reuse the venv** in a fresh shell, after it has been created and deps
installed, just run the `activate` command above.

## Running

You can run the code using the `mbfte.py` command-line program. For example,
run the following to print useful help information.

    > python mbfte.py --help

In order to run the script on anything useful, you'll need a `pytorch` model.
You can use the default pretrained model by passing `--model-dir PRETRAINED`
(this is the default).

## Examples

### Benchmarking

The following command runs 10 iterations of the full encode-decode cycle for
plaintexts of length 10.

    > python mbfte.py benchmark --length 10 --ntrials 10

### Encoding

The following command encodes a message using a "random" key and no random
padding. The key is a hex-encoded 256-bit key.

    > python mbfte.py --padding 0 encode \
        "c960efce6667ec5ca16851425a4619aa096ec8cb143d83eac3d4fc271be3a626" \
        --plaintext "secret message"


### Decoding

The following command decodes the covertext produced by the above command.

    > python mbfte.py --padding 0 decode \
        "c960efce6667ec5ca16851425a4619aa096ec8cb143d83eac3d4fc271be3a626" \
        --covertext " SVG Twitch streamed CCT incredible debut \"The World Chosing Prevention: How Lethal"


### Encoding + Decoding

The following command runs the full encode-decode cycle using a "random" key and
no random padding.

    > python mbfte.py --padding 0 encode-decode "secret message" \
        "c960efce6667ec5ca16851425a4619aa096ec8cb143d83eac3d4fc271be3a626"


## Internal details

At a high level, the MB-FTE algorithm we use can be explained as follows. Given
some plaintext message, we first encrypt this using our symmetric key encryption
scheme and then "encode" this ciphertext into text. This encoding works as
follows. Starting from some seed text, we retrieve (token, probability) pairs
from the language model given as a cumulative probability distribution. We then
map bits of the ciphertext to a probability, and select the token that falls
within this probability bound. Because we need this mapping to be reversable (in
order for decoding to succeed), we utilize arithmetic coding in order to do this
mapping.

With this algorithm in mind, the codebase is split up as follows:

- `crypto.py`: The symmetric key encryption scheme(s) used.
- `model_wrapper.py`: An abstract class for writing so-called "model wrappers".
  These present an API in which one can request a model prediction given a token
  and some prior model state, getting a cumulative probability distribution over
  the predicted tokens.
- `pytorch_model_wrapper.py`: An implementation of the model wrapper for
  `pytorch`. This actually exposes three model wrappers:
  - `PyTorchModelWrapper`: The "base" model wrapper that simply calls `pytorch`
    and returns the cumulative probability distribution as emitted by `pytorch`.
  - `TopKModelWrapper`: A wrapper around `PyTorchModelWrapper` that
    post-processes the emitted cumulative probability distribution to only emit
    the top `k` tokens (for some user-specified `k`).
  - `VariableTopKModelWrapper`: A wrapper around `PyTorchModelWrapper` that
    post-processes the emitted cumulative probability distribution to only emit
    the top `k` tokens, where `k` is selected based on some precision threshold.
    That is, `k` is chosen on each iteration to correspond to the largest token
    such that the difference between its probability and the prior token's
    probability is greater than some precision threshold.
- `textcover.py`: The core `TextCover` class, which combines a symmetric key
  scheme and model wrapper to implement the MB-FTE encode and decode operations.
- `utils.py`: Helpful utility functions.

## Running on an ARM64 docker container

To run on the provided ARM64 docker container, you'll need to install `qemu`,
and in particular, the following three packages: `qemu`, `binfmt-support`,
`qemu-user-static`.

Once installed, run the following `docker` command to set up `qemu`:

    > docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

Lastly, you can build the ARM64 docker container by running the following:

    > docker build --file ./Dockerfile_arm64 -t "arm64v8/ubuntu:mbfte" .

This will build all the necessary dependencies and set up the container to run
the code.

To enter a bash shell in the built container, run the following:

    > docker run -v $PWD:/usr/src/MB-FTE -it arm64v8/ubuntu:mbfte /bin/bash

This will mount the python code in the container and launch a shell.

## Testing cross platform compatibility

In order to test cross platform compatibility, run the following.

On your local (non-ARM64) host, run the `benchmark` command, saving the
covertexts to a file:

    (HOST)> python mbfte.py benchmark --ntrials 100 --save-covertexts covertexts.txt

The `covertexts.txt` file contains all the generated covertexts, alongside a
comment at the top of the file detailed the specifications of the benchmark.
This includes the key used. Copy this value, we'll need it below.

On your ARM64 docker container, run the `decode` command, using the saved
covertexts as the strings to decode alongside the copied key value:

    (ARM64)> python mbfte.py decode --covertext-file covertexts.txt <key>

## Cross platform compatibility status

As of 11/16/23, we have the following results for various cross-platform fix
attempts. All of these are the result of running between a Linux x86-64 host and
the ARM64 docker container.

| CLI arguments for potential x-platform "fix"                                                             | Status |
| -------------------------------------------------------------------------------------------------------- | :----: |
| `--model-params '{"float64": true}'` (the default)                                                       |   ✅    |
| `--model-params '{"float64": false}' --model-type variable-top-k --precision 3 --extra-encoding-bits 20` |   ❌    |
| `--model-params '{"float64": false}' --model-type variable-top-k --precision 2 --extra-encoding-bits 40` |   ❌    |

In summary, 64-bit floats appear to work across the board. As soon as we move to
32-bit floats, we have potential issues.

## Documentation

You can build and view available documentation as follows:

    > cd docs
    > make html
    > <your-browser-of-choice> _build/html/index.html

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## Authors

This implementation was developed as a joint collaboration between University of
Florida and Galois, Inc. The following people have contributed to the
development:

- Luke Bauer
- Himanshu Goyal
- Alex Grushin
- Chris Phifer
- Alex J Malozemoff
- Hari Menon

## Acknowledgments

This material is based upon work supported by the Defense Advanced Research
Projects Agency (DARPA) under Contract Number FA8750-19-C-0085. Any opinions,
findings and conclusions or recommendations expressed in this material are those
of the author(s) and do not necessarily reflect the views of the DARPA.

Distribution Statement "A" (Approved for Public Release, Distribution
Unlimited)

Copyright © 2019-2023 Galois, Inc.
