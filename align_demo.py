import os
import re
import json
import difflib
import warnings
import whisperx

# Silence harmless PyTorch / Pyannote / Lightning console warnings
warnings.filterwarnings("ignore")

# 1. Portable relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FILE = os.path.join(BASE_DIR, "audio_output", "introduction.mp3")
MANUSCRIPT_FILE = os.path.join(BASE_DIR, "manuscripts", "introduction.txt")
OUTPUT_JSON = os.path.join(BASE_DIR, "audio_output", "introduction_cues.json")

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BATCH_SIZE = 4

def normalize_word(w: str) -> str:
    """Normalizes a word for acoustic-to-manuscript comparison."""
    t = w.lower()
    # Map Serbian diacritics and Latin diphthongs to ASCII equivalents
    replacements = {
        'ž': 'z', 'č': 'c', 'ć': 'c', 'š': 's', 'đ': 'dj', 'oe': 'e', 'ae': 'e'
    }
    for src, dst in replacements.items():
        t = t.replace(src, dst)
    return re.sub(r'[^\w]', '', t)

# 2. Load ground-truth manuscript text and split into sentences & words
with open(MANUSCRIPT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read().strip()

sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text) if s.strip()]

ms_words_norm = []
sentence_word_ranges = []  # Stores (first_word_idx, last_word_idx) for each sentence

for sent in sentences:
    # Split on whitespace so contractions like "city's" remain a single word token
    raw_tokens = [w for w in sent.split() if normalize_word(w)]
    start_i = len(ms_words_norm)
    for w in raw_tokens:
        ms_words_norm.append(normalize_word(w))
    end_i = len(ms_words_norm) - 1
    sentence_word_ranges.append((sent, start_i, end_i))

# 3. Transcribe & Force-Align Audio with WhisperX
print(f"Loading audio from: {AUDIO_FILE}")
audio = whisperx.load_audio(AUDIO_FILE)
audio_duration = round(len(audio) / 16000.0, 2)

print("Loading Whisper transcription model...")
model = whisperx.load_model("small.en", DEVICE, compute_type=COMPUTE_TYPE)

print("Transcribing...")
result = model.transcribe(audio, batch_size=BATCH_SIZE)

print("Loading alignment model...")
align_model, metadata = whisperx.load_align_model(
    language_code=result["language"], 
    device=DEVICE
)

print("Aligning audio words...")
aligned_result = whisperx.align(
    result["segments"], 
    align_model, 
    metadata, 
    audio, 
    DEVICE, 
    return_char_alignments=False
)

# 4. Extract flat list of timestamped audio words
audio_words = []
for seg in aligned_result.get("segments", []):
    for w in seg.get("words", []):
        if "start" in w and "end" in w and normalize_word(w.get("word", "")):
            audio_words.append({
                "norm": normalize_word(w["word"]),
                "start": float(w["start"]),
                "end": float(w["end"])
            })

audio_words_norm = [w["norm"] for w in audio_words]

# 5. Global Opcode Alignment between Manuscript Words and Audio Words
matcher = difflib.SequenceMatcher(None, ms_words_norm, audio_words_norm, autojunk=False)

ms_word_start = [0.0] * len(ms_words_norm)
ms_word_end = [0.0] * len(ms_words_norm)

for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    if tag == "equal":
        for k in range(i2 - i1):
            ms_word_start[i1 + k] = audio_words[j1 + k]["start"]
            ms_word_end[i1 + k] = audio_words[j1 + k]["end"]
    elif tag == "replace":
        # Map the replaced manuscript word span across the corresponding audio word span
        span_start = audio_words[j1]["start"]
        span_end = audio_words[j2 - 1]["end"]
        count = i2 - i1
        dur = (span_end - span_start) / max(count, 1)
        for k in range(count):
            ms_word_start[i1 + k] = span_start + k * dur
            ms_word_end[i1 + k] = span_start + (k + 1) * dur
    elif tag == "delete":
        # Manuscript word was skipped in audio; anchor to neighboring timestamp
        fallback_t = audio_words[j1]["start"] if j1 < len(audio_words) else (audio_words[-1]["end"] if audio_words else 0.0)
        for i in range(i1, i2):
            ms_word_start[i] = fallback_t
            ms_word_end[i] = fallback_t

# 6. Build sentence cues with exact manuscript text and locked boundaries
cues = []
for sent_text, start_i, end_i in sentence_word_ranges:
    cues.append({
        "start": round(ms_word_start[start_i], 2),
        "end": round(ms_word_end[end_i], 2),
        "text": sent_text
    })

# Clamp final sentence end to audio duration if needed
if cues and cues[-1]["end"] < cues[-1]["start"]:
    cues[-1]["end"] = audio_duration

# 7. Save to JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(cues, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 50)
print(f"Alignment complete! Generated {len(cues)} sentence cues.")
print(f"Saved output to: {OUTPUT_JSON}")
print("=" * 50)