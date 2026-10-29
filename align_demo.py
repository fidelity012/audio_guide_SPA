import os
import json
import re
import whisperx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FILE = os.path.join(BASE_DIR, "audio_output", "introduction.mp3")
MANUSCRIPT_FILE = os.path.join(BASE_DIR, "manuscripts", "introduction.txt")
OUTPUT_JSON = os.path.join(BASE_DIR, "audio_output", "introduction_cues.json")

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BATCH_SIZE = 4

# 1. Read ground-truth manuscript sentences for display
with open(MANUSCRIPT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split into clean, non-empty sentences
display_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text) if s.strip()]

# 2. Transcribe & Align audio
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

print("Aligning audio...")
aligned_result = whisperx.align(
    result["segments"], 
    align_model, 
    metadata, 
    audio, 
    DEVICE, 
    return_char_alignments=False
)

# 3. Collect all aligned words with valid timestamps
aligned_words = []
for seg in aligned_result.get("segments", []):
    for w in seg.get("words", []):
        if "start" in w and "end" in w:
            aligned_words.append(w)

# 4. Map timestamps back onto the authentic manuscript sentences
cues = []
word_idx = 0
total_words = len(aligned_words)

for sent in display_sentences:
    # Count words in current manuscript sentence
    sent_word_count = len(re.findall(r'\b\w+\b', sent))
    
    if word_idx < total_words:
        start_time = aligned_words[word_idx]["start"]
        end_idx = min(word_idx + sent_word_count - 1, total_words - 1)
        end_time = aligned_words[end_idx]["end"]
        word_idx += sent_word_count
    else:
        # Fallback if audio ends slightly early
        start_time = cues[-1]["end"] if cues else 0.0
        end_time = start_time + 2.0

    cues.append({
        "start": round(start_time, 2),
        "end": round(end_time, 2),
        "text": sent
    })

# 5. Save the output
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(cues, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 50)
print(f"Alignment complete! Generated {len(cues)} sentence cues.")
print(f"Saved output to: {OUTPUT_JSON}")
print("=" * 50)