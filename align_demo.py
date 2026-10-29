import os
import json
import whisperx

# 1. Portable relative paths matching your project setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FILE = os.path.join(BASE_DIR, "audio_output", "introduction.mp3")
MANUSCRIPT_FILE = os.path.join(BASE_DIR, "manuscripts", "introduction.txt")
OUTPUT_JSON = os.path.join(BASE_DIR, "audio_output", "introduction_cues.json")

# 2. Configuration: CPU inference
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BATCH_SIZE = 4

print(f"Loading audio from: {AUDIO_FILE}")
audio = whisperx.load_audio(AUDIO_FILE)

print("Loading Whisper transcription model...")
model = whisperx.load_model("small.en", DEVICE, compute_type=COMPUTE_TYPE)

print("Transcribing...")
result = model.transcribe(audio, batch_size=BATCH_SIZE)

print("Loading alignment model...")
align_model, metadata = whisperx.load_align_model(
    language_code=result["language"], 
    device=DEVICE
)

print("Aligning audio to transcript...")
aligned_result = whisperx.align(
    result["segments"], 
    align_model, 
    metadata, 
    audio, 
    DEVICE, 
    return_char_alignments=False
)

# 3. Format cues to clean sentence start/end timestamps
cues = []
for segment in aligned_result["segments"]:
    cues.append({
        "start": round(segment["start"], 2),
        "end": round(segment["end"], 2),
        "text": segment["text"].strip()
    })

# 4. Save to JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(cues, f, ensure_ascii=False, indent=2)

print("\n" + "="*50)
print(f"Alignment complete! Generated {len(cues)} sentence cues.")
print(f"Saved output to: {OUTPUT_JSON}")
print("="*50)