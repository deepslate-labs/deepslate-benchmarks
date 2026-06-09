import os
import json
import tempfile
from pathlib import Path
import soundfile as sf
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# --- CONFIGURATION ---
# Directory containing your pipeline's output .wav files
BENCH_DIR = Path(__file__).resolve().parent
RESPONSE_DIR = BENCH_DIR / "benchmark_outputs"
# Pattern for your files. {id} will be replaced by the dataset ID (0-999)
RESPONSE_FILE_PATTERN = "response_{id}.wav"
# Output file for detailed evaluation results
RESULTS_PATH = BENCH_DIR / "bba_evaluation_results.json"
# Dataset split to evaluate

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Anthropic API key for the Claude judge (direct first-party API).
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Models
# OpenAI model to transcribe the responses
TRANSCRIPTION_MODEL = "whisper-1"
# Anthropic model for the judge.
# The original Artificial Analysis Big Bench Audio methodology judged with
# Claude 3.5 Sonnet, but that model is retired on the direct Anthropic API (404
# since 2025-10-28). claude-sonnet-4-6 is its drop-in replacement and keeps the
# judge in the same Sonnet family. Note: a different judge than the original
# means scores are not directly comparable to the published leaderboard.
JUDGE_MODEL = "claude-sonnet-4-6"

# --- CLIENT SETUP ---
def build_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)


def build_anthropic_client():
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency for the Claude judge. Install anthropic to continue."
        ) from e
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY must be set for the Claude judge.")
    return Anthropic(api_key=ANTHROPIC_API_KEY)


def transcribe_audio(openai_client, audio_path_or_file):
    """Transcribes audio using OpenAI Whisper."""
    try:
        with open(audio_path_or_file, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"Error transcribing {audio_path_or_file}: {e}")
        return ""


def _extract_text(response):
    if not response or not getattr(response, "content", None):
        return ""
    parts = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts).strip()


def get_judge_decision(anthropic_client, question_text, official_answer, candidate_answer):
    """
    Calls the LLM Judge (Claude via the direct Anthropic API) to evaluate the
    answer using the exact prompt from the Artificial Analysis methodology.
    """
    prompt_template = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.
If the CANDIDATE ANSWER contradicts itself, assess the first proposed answer.
If the CANDIDATE ANSWER provides a final answer and working, assess the final answer only.
If the CANDIDATE ANSWER includes irrelevant information, assess only the relevant information.
If the CANDIDATE ANSWER includes a numeric value it is ok if it is spelled e.g. 7 or seven
It is ok if the CANDIDATE ANSWER involves a misspelling of a person's name e.g. Leda or Lida, Autry or Audrie.
The question, for reference only: START QUESTION {question}

END QUESTION
The OFFICIAL ANSWER:{official_answer}
BEGIN CANDIDATE ANSWER TO ASSESS {candidate_answer}
END CANDIDATE ANSWER TO ASSESS
Reply only with CORRECT or INCORRECT."""

    formatted_prompt = prompt_template.format(
        question=question_text,
        official_answer=official_answer,
        candidate_answer=candidate_answer
    )

    try:
        response = anthropic_client.messages.create(
            model=JUDGE_MODEL,
            temperature=0,
            max_tokens=10,
            messages=[{"role": "user", "content": formatted_prompt}],
        )
        return _extract_text(response)
    except Exception as e:
        print(f"Error calling Claude judge: {e}")
        return "ERROR"


def main():
    openai_client = build_openai_client()
    anthropic_client = build_anthropic_client()

    # 1. Load the Benchmark Dataset
    print("Loading Big Bench Audio dataset...")
    dataset = load_dataset("ArtificialAnalysis/big_bench_audio", split="train")

    results = []
    correct_count = 0
    total_count = 0
    missing_count = 0

    # 2. Iterate through the dataset
    print(f"Starting evaluation of {len(dataset)} items...")

    for item in tqdm(dataset):
        item_id = item['id']
        category = item['category']
        official_answer = item['official_answer']

        # Locate user's response file
        response_filename = RESPONSE_FILE_PATTERN.format(id=item_id)
        response_path = RESPONSE_DIR / response_filename

        if not os.path.exists(response_path):
            # Skip if file missing
            missing_count += 1
            continue

        # 3. Get Question Text
        # The dataset contains audio. We must transcribe the INPUT question first
        # to provide context to the judge.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_input:
            # Save input audio from dataset to temp file for Whisper
            input_audio_array = item['audio']['array']
            sample_rate = item['audio']['sampling_rate']
            sf.write(temp_input.name, input_audio_array, sample_rate)

            question_text = transcribe_audio(openai_client, temp_input.name)

        # 4. Get Candidate Transcript
        # Transcribe the user's generated response
        candidate_transcript = transcribe_audio(openai_client, response_path)

        # 5. Judge
        if not question_text or not candidate_transcript:
            decision = "ERROR: transcription_failed"
            is_correct = False
        else:
            decision = get_judge_decision(
                anthropic_client,
                question_text,
                official_answer,
                candidate_transcript,
            )
            decision_normalized = decision.strip().upper().strip(".")
            is_correct = decision_normalized == "CORRECT"
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "id": item_id,
            "category": category,
            "question_transcript": question_text,
            "candidate_transcript": candidate_transcript,
            "official_answer": official_answer,
            "judge_output": decision,
            "is_correct": is_correct
        })

    # 6. Calculate and Print Stats
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"\n--- Final Results ---")
        print(f"Total Evaluated: {total_count}")
        if missing_count:
            print(f"Missing Responses: {missing_count}")
        print(f"Accuracy: {accuracy:.2f}%")

        # Save detailed results
        results_dir = RESULTS_PATH.parent
        os.makedirs(results_dir, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {RESULTS_PATH}")
    else:
        print("No files were evaluated. Check RESPONSE_DIR and RESPONSE_FILE_PATTERN.")


if __name__ == "__main__":
    main()
