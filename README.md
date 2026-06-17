> [!Note]
> Currently Deepslate's Speech Generation is still in closed beta, therefore these Benchmarks can only be publicly validated with external TTS Models like Elevenlabs for now. Changes to this repository will be made once it is in a public preview. 
>
> Due to some comments: This does not mean that Deepslate Opal is not capable of true native Speech to Speech. Opal understands Audio natively without Text representation and is able to output either Text or Audio directly. As long as the Speech Generation is in closed beta, the public API will run the Model in Audio in & Text out Mode. For more details see [here](https://docs.deepslate.eu/opal#speech-to-text:~:text=Speech%2Dto%2DText,for%20voice%20generation.).

# Deepslate Benchmarks

Benchmark suite for Deepslate speech model Opal using the realtime WebSocket API.

## Resources

- API docs: https://docs.deepslate.eu/websocket

## Benchmarks

- `s2s/big-bench-audio`: Speech-to-speech QA on Big Bench Audio.

Each benchmark folder has its own README with dataset details and evaluation notes.

## Requirements

- Python 3.12+
- Deepslate API credentials (see below)
- Additional evaluation credentials depending on the benchmark

## Install

```shell
python -m venv .venv
source .venv/bin/activate

pip install -e .
```

The realtime protocol is driven through the official [`deepslate-core`](https://pypi.org/project/deepslate-core/)
SDK, which ships and versions the protobuf bindings — no local proto generation
required.

## Configure Deepslate credentials

Set these environment variables before running benchmarks:

- `DEEPSLATE_API_KEY`
- `DEEPSLATE_VENDOR_ID`
- `DEEPSLATE_ORG_ID`

## Quick start

```shell
export DEEPSLATE_API_KEY="your_api_key"
export DEEPSLATE_VENDOR_ID="your_vendor_id"
export DEEPSLATE_ORG_ID="your_org_id"

python s2s/big-bench-audio/run_benchmark.py
```

## Benchmark structure

Most benchmarks follow the same flow:

- `run_benchmark.py` runs the benchmark and writes outputs (for example, audio files or JSON logs).
- `evaluate.py` reads those outputs and produces metrics or scores (when provided).

Refer to each benchmark README for exact commands, required env vars, and any extra
Python dependencies needed for evaluation.

## Evaluate results

Evaluation is benchmark-specific and may require additional credentials. Example:

```shell
python s2s/big-bench-audio/evaluate.py
```

## Outputs

Outputs are written to `benchmark_outputs/` inside each benchmark folder unless
stated otherwise. Common output types include audio responses, transcripts, and
JSON results files.
