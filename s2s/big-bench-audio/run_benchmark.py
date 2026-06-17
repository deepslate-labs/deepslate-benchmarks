import asyncio
import os
import time
import wave
from pathlib import Path

import aiohttp
import librosa
import numpy as np
from datasets import load_dataset

from deepslate.core import (
    DeepslateOptions,
    DeepslateSession,
    DeepslateSessionListener,
    ElevenLabsTtsConfig,
    HostedTtsConfig,
    TriggerMode,
    VadConfig,
)

# --- CONFIGURATION ---
BENCH_DIR = Path(__file__).resolve().parent
# DeepslateOptions.from_env() reads DEEPSLATE_ORGANIZATION_ID; this repo's docs
# use DEEPSLATE_ORG_ID, so accept both.
ORG_ID = os.getenv("DEEPSLATE_ORGANIZATION_ID") or os.getenv("DEEPSLATE_ORG_ID")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR") or (BENCH_DIR / "benchmark_outputs"))

# Base URL override, e.g. https://app.staging.deepslate.eu for staging. The SDK
# defaults to production (https://app.deepslate.eu) when this is unset.
BASE_URL = os.getenv("DEEPSLATE_BASE_URL")
WS_URL = os.getenv("DEEPSLATE_WS_URL")  # direct websocket endpoint (bypasses gateway)
SYSTEM_PROMPT = os.getenv("DEEPSLATE_SYSTEM_PROMPT", "You are a helpful assistant.")
TEMPERATURE = float(os.getenv("DEEPSLATE_TEMPERATURE", "0"))

# TTS provider: "hosted" (Deepslate-hosted/cloned voice, default) or "elevenlabs".
TTS_PROVIDER = os.getenv("DEEPSLATE_TTS_PROVIDER", "hosted")
# Deepslate-hosted voice id (used when TTS_PROVIDER == "hosted").
HOSTED_TTS_VOICE_ID = os.getenv(
    "DEEPSLATE_TTS_VOICE_ID", "c3dfa73f-a1ab-4aad-b48a-0e9b9fe4a69f"
)
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
ELEVEN_LABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# Audio rate sent to / received from Deepslate. The SDK ties the output line to
# the input line, so one rate governs both. 16kHz is the standard speech rate
# and a supported TTS output rate; 48kHz output makes the server terminate the
# session, so keep this at 16000.
SAMPLE_RATE = 16000

# Optional: cap how many items to process (handy for smoke tests). 0 = all.
LIMIT = int(os.getenv("LIMIT", "0"))

# Input pacing: 1.0 = stream at real time (like a live mic). The server needs
# the utterance to be processed at roughly real time before flush_vad commits
# it; bursting the audio (PACE=0) makes the model only "hear" a fragment and
# answer the wrong question, so keep this at 1.0 for correct results.
PACE = float(os.getenv("PACE", "1.0"))

# Number of questions to process concurrently (independent websocket sessions).
# Real-time input pacing dominates wall-clock, so concurrency is how a full
# 1000-item run finishes in hours instead of half a day.
CONCURRENCY = int(os.getenv("CONCURRENCY", "6"))

# Completion tuning.
# The realtime server does not reliably emit ResponseEnd in this turn-based
# flow, but it does deliver the full audio. We therefore treat a response as
# complete once no audio chunk has arrived for RESPONSE_SILENCE_GRACE seconds
# (after at least one chunk), or when ResponseEnd does arrive.
FIRST_CHUNK_TIMEOUT = float(os.getenv("FIRST_CHUNK_TIMEOUT", "25"))      # max wait for the response to start
RESPONSE_SILENCE_GRACE = float(os.getenv("RESPONSE_SILENCE_GRACE", "3"))  # quiet gap after last audio chunk => done
HARD_TIMEOUT = float(os.getenv("HARD_TIMEOUT", "180"))                    # absolute ceiling per question (backstop only)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_options():
    # Big Bench Audio is a reasoning QA benchmark with deterministic official
    # answers, so decode greedily for a representative, reproducible score.
    # (The SDK defaults temperature to 1.0, which makes the model ramble and
    # hurts the judged answer.) temperature=0.0 is a proto3 default, so it is
    # omitted on the wire and the server decodes deterministically.
    if WS_URL:
        return DeepslateOptions(
            vendor_id=os.getenv("DEEPSLATE_VENDOR_ID", "x"),
            organization_id=ORG_ID or "x",
            api_key=os.getenv("DEEPSLATE_API_KEY", ""),
            ws_url=WS_URL,
            system_prompt=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
        )
    kwargs = dict(
        organization_id=ORG_ID,
        system_prompt=SYSTEM_PROMPT,
        temperature=TEMPERATURE,
    )
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return DeepslateOptions.from_env(**kwargs)


def build_vad_config():
    return VadConfig(
        confidence_threshold=0.5,
        min_volume=0.0,
        start_duration_ms=300,
        stop_duration_ms=700,
        backbuffer_duration_ms=1000,
    )


def build_tts_config():
    if TTS_PROVIDER == "elevenlabs":
        return ElevenLabsTtsConfig(
            api_key=ELEVEN_LABS_API_KEY,
            voice_id=ELEVEN_LABS_VOICE_ID,
        )
    # Deepslate-hosted (cloned) voice — no external TTS credentials needed.
    return HostedTtsConfig(voice_id=HOSTED_TTS_VOICE_ID)


class _ResponseCollector(DeepslateSessionListener):
    """Collects a single model turn's audio and tracks completion timing."""

    def __init__(self):
        self.ready = asyncio.Event()
        self.audio_chunks = []
        self.transcripts = []
        self.first_chunk_seen = False
        self.first_audio_time = None
        self.last_audio_time = None
        self.response_ended = False
        self.out_sample_rate = SAMPLE_RATE
        self.out_channels = 1

    async def on_session_initialized(self):
        self.ready.set()

    async def on_audio_chunk(self, pcm_bytes, sample_rate, channels, transcript):
        self.audio_chunks.append(pcm_bytes)
        if transcript:
            self.transcripts.append(transcript)
        now = time.monotonic()
        if self.first_audio_time is None:
            self.first_audio_time = now
        self.first_chunk_seen = True
        self.last_audio_time = now
        self.out_sample_rate = sample_rate
        self.out_channels = channels

    async def on_playback_buffer_clear(self):
        # User-start / barge-in marker; clear anything buffered so far so we only
        # keep the actual model turn.
        if not self.first_chunk_seen:
            self.audio_chunks = []

    async def on_response_end(self, turn_id=0):
        self.response_ended = True

    async def on_error(self, category, message, trace_id):
        print(f"Deepslate error [{category}]: {message}")


async def run_single_inference(audio_array, sample_rate, question_id, http_session):
    """
    Sends one audio question to Deepslate and captures the audio response.

    Flow: stream the utterance with NO_TRIGGER, then commit it with
    trigger_inference(flush_vad=True) to trigger exactly one inference.
    """
    # Resample audio to the session rate, mono.
    if sample_rate != SAMPLE_RATE:
        audio_array = librosa.resample(
            y=audio_array, orig_sr=sample_rate, target_sr=SAMPLE_RATE
        )

    # Convert float32 [-1, 1] to int16 PCM.
    pcm_data = (audio_array * 32767).astype(np.int16).tobytes()

    listener = _ResponseCollector()
    session = DeepslateSession.create(
        build_options(),
        vad_config=build_vad_config(),
        tts_config=build_tts_config(),
        user_agent="deepslate-benchmarks/big-bench-audio",
        http_session=http_session,
        listener=listener,
    )
    # Disable the SDK's mid-turn reconnect: a one-shot turn can't survive a
    # reconnect (the buffered utterance is dropped), so we retry at the
    # process_item level instead.
    session._options.max_retries = 0
    session.start()

    try:
        # Wait for the session to be ready before streaming so the paced audio
        # is processed in real time rather than flushed in a burst. initialize()
        # is a no-op until the background task has connected the websocket, so
        # poll it until SessionReady arrives.
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 30.0
        while not listener.ready.is_set() and loop.time() < deadline:
            await session.initialize(sample_rate=SAMPLE_RATE, channels=1)
            try:
                await asyncio.wait_for(listener.ready.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass
        if not listener.ready.is_set():
            raise TimeoutError("session did not initialize within 30s")

        # Stream Audio Chunks (NO_TRIGGER: just feed the utterance).
        chunk_size = 2 * SAMPLE_RATE // 10  # 100ms of 16-bit mono
        chunk_start = time.monotonic()
        sent_chunks = 0
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i + chunk_size]
            await session.send_audio(
                chunk, SAMPLE_RATE, 1, trigger=TriggerMode.NO_TRIGGER
            )
            sent_chunks += 1
            # Optionally pace sending to (a fraction of) real-time audio duration.
            if PACE > 0:
                chunk_duration = len(chunk) / (2 * SAMPLE_RATE)
                target_elapsed = sent_chunks * chunk_duration * PACE
                sleep_for = target_elapsed - (time.monotonic() - chunk_start)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

        # Commit the buffered speech and trigger a single inference.
        trigger_time = time.monotonic()
        await session.trigger_inference(flush_vad=True)

        # Keep feeding silence so the input pipeline stays alive while the model
        # responds (mirrors a client holding the mic open).
        response_complete = False

        async def send_silence_until_response():
            silence_chunk = b"\x00" * chunk_size
            silence_duration = len(silence_chunk) / (2 * SAMPLE_RATE)
            while not response_complete:
                await session.send_audio(
                    silence_chunk, SAMPLE_RATE, 1, trigger=TriggerMode.NO_TRIGGER
                )
                await asyncio.sleep(silence_duration)

        silence_task = asyncio.create_task(send_silence_until_response())

        # Wait for the response, polling the listener's timing state.
        recv_start = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                if listener.response_ended:
                    break
                if listener.last_audio_time is not None:
                    if now - listener.last_audio_time >= RESPONSE_SILENCE_GRACE:
                        # Quiet gap after audio => response finished.
                        break
                elif now - recv_start >= FIRST_CHUNK_TIMEOUT:
                    print(f"Question {question_id}: no response started; giving up.")
                    return None
                if now - recv_start >= HARD_TIMEOUT:
                    break
                await asyncio.sleep(0.1)
        finally:
            response_complete = True
            silence_task.cancel()
            try:
                await silence_task
            except asyncio.CancelledError:
                pass

        ttfa = (
            listener.first_audio_time - trigger_time
            if listener.first_audio_time is not None
            else None
        )
        return b"".join(listener.audio_chunks), listener.out_sample_rate, "".join(listener.transcripts), ttfa
    finally:
        await session.close()


async def process_item(item, idx, total, sem, progress, http_session):
    question_id = item['id']
    output_filename = OUTPUT_DIR / f"response_{question_id}.wav"

    # Resume support: skip questions we've already captured.
    if output_filename.exists():
        return

    audio_array = item['audio']['array']
    sr = item['audio']['sampling_rate']
    request_audio_duration = len(audio_array) / sr

    async with sem:
        progress['started'] += 1
        print(
            f"[{progress['done']}/{total} done] start {idx} (ID: {question_id}) "
            f"[req {request_audio_duration:.1f}s]"
        )

        # Run inference, retrying transient connection failures.
        result = None
        for attempt in range(3):
            try:
                result = await run_single_inference(
                    audio_array, sr, question_id, http_session
                )
                break
            except Exception as e:  # network hiccup etc. — retry then skip
                print(f"Question {question_id}: error {e!r} (attempt {attempt + 1}/3)")
                await asyncio.sleep(2)
        else:
            print(f"Question {question_id}: giving up after retries; skipping.")
            progress['done'] += 1
            return

    progress['done'] += 1

    if result is None:
        print(f"  Q{question_id}: no response (timeout); not saved.")
        return
    response_audio, out_sample_rate, transcript, ttfa = result
    if not response_audio:
        print(f"  Q{question_id}: empty response; not saved.")
        return

    with wave.open(os.fspath(output_filename), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(out_sample_rate)
        wav_file.writeframes(response_audio)
    # Sidecar: the model's own transcript of the spoken answer (may be empty if
    # the TTS path doesn't surface it) — lets us compare against Whisper.
    (OUTPUT_DIR / f"transcript_{question_id}.txt").write_text(transcript or "")
    # Sidecar: time-to-first-audio (seconds) for the latency metrics in the report.
    (OUTPUT_DIR / f"ttfa_{question_id}.txt").write_text("" if ttfa is None else f"{ttfa:.4f}")
    dur = len(response_audio) / 2 / out_sample_rate
    print(f"  saved {output_filename.name} ({dur:.1f}s, transcript {len(transcript)} chars)")


async def amain():
    print("Loading Big Bench Audio...")
    dataset = load_dataset("ArtificialAnalysis/big_bench_audio", split="train")
    total = len(dataset)

    items = list(enumerate(dataset))
    if LIMIT:
        # Only count items that still need processing toward the limit.
        pending = [
            (i, it) for (i, it) in items
            if not (OUTPUT_DIR / f"response_{it['id']}.wav").exists()
        ]
        items = pending[:LIMIT]

    sem = asyncio.Semaphore(CONCURRENCY)
    progress = {'started': 0, 'done': 0}
    print(f"Processing up to {len(items)} items with concurrency {CONCURRENCY}, pace {PACE}x real-time.")
    async with aiohttp.ClientSession() as http_session:
        tasks = [
            asyncio.create_task(
                process_item(it, i, total, sem, progress, http_session)
            )
            for (i, it) in items
        ]
        await asyncio.gather(*tasks)
    print("All done.")


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
