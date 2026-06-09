# Big Bench Audio Benchmark

Run Big Bench Audio QA through the Deepslate realtime API and evaluate responses with Whisper transcription plus a
Claude judge via the direct Anthropic API.

## References

* https://huggingface.co/blog/big-bench-audio-release

## Requirements

- Install repo dependencies (see root `README.md`).
- Deepslate credentials:
    - `DEEPSLATE_API_KEY`
    - `DEEPSLATE_VENDOR_ID`
    - `DEEPSLATE_ORG_ID`
- ElevenLabs TTS credentials:
    - `ELEVEN_LABS_API_KEY`
- Evaluation credentials:
    - `OPENAI_API_KEY`
    - `ANTHROPIC_API_KEY`

## Run benchmark

```shell
export DEEPSLATE_API_KEY="your_api_key"
export DEEPSLATE_VENDOR_ID="your_vendor_id"
export DEEPSLATE_ORG_ID="your_org_id"
export ELEVEN_LABS_API_KEY="your_eleven_labs_api_key"

python run_benchmark.py
```

## Evaluate

```shell
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_anthropic_key"

python evaluate.py
```

## Outputs

- Audio responses: `benchmark_outputs/response_{id}.wav` (this folder)
- Detailed results: `bba_evaluation_results.json` (this folder)
