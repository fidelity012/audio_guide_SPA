import os
import re
import json
import shutil
import difflib
import warnings
import argparse
import subprocess
import numpy as np
import soundfile as sf
from kokoro import KPipeline
import whisperx

# =====================================================================
# 0. ENVIRONMENT & WARNING SUPPRESSION
# =====================================================================
# Silence PyTorch, HF Hub, TorchCodec, and Lightning console notices
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANUSCRIPTS_DIR = os.path.join(BASE_DIR, "manuscripts")
OUTPUT_DIR = os.path.join(BASE_DIR, "audio_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 24000
BITRATE = "96k"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_BATCH_SIZE = 4

# =====================================================================
# 1. PHONETIC & NUMERICAL NORMALIZATION (from gui_tts.py)
# =====================================================================
ROMAN_ORDINALS = {
    'I': 'First', 'II': 'Second', 'III': 'Third', 'IV': 'Fourth',
    'V': 'Fifth', 'VI': 'Sixth', 'VII': 'Seventh', 'VIII': 'Eighth',
    'IX': 'Ninth', 'X': 'Tenth', 'XI': 'Eleventh', 'XII': 'Twelfth',
    'XIII': 'Thirteenth', 'XIV': 'Fourteenth', 'XV': 'Fifteenth',
    'XVI': 'Sixteenth', 'XVII': 'Seventeenth', 'XVIII': 'Eighteenth',
    'XIX': 'Nineteenth', 'XX': 'Twentieth'
}

CUSTOM_SERBIAN_WORDS = {
    "belovode", "kostolac", "vlasac", "padina", "dubravica", "drmno",
    "golubac", "dubovac", "klenovnik", "usije", "salakovac", "trnjane",
    "kasidol", "klenje", "leskovac", "tabanovac", "krvije", "stubline",
    "drenovac", "vatin", "gava", "brdo",
    "morava", "mlava", "busur", "beljanica",
    "krstic", "jacanovic", "sljivar"
}

SPECIAL_TERMS = {
    "moesia": "[Moesia](/miːʃə/)",
    "margum": "[Margum](/mɑːrɡʊm/)",
    "lederata": "[Lederata](/lɛdɛrɑːtɑː/)",
    "pincum": "[Pincum](/pɪŋkʊm/)",
    "cuppae": "[Cuppae](/kʊpaɪ/)",
    "novae": "[Novae](/noʊvaɪ/)",
    "kaliste": "[Kalište](/kɑːliːʃtɛ/)",
    "kalište": "[Kalište](/kɑːliːʃtɛ/)",
    "mansio": "[Mansio](/mɑːnsiːoʊ/)",
    "pagus": "[Pagus](/pɑːɡʊs/)",
    "castrum": "[castrum](/kɑːstrʊm/)",
    "castra": "[castra](/kɑːstrɑː/)",
    "canabae": "[canabae](/kɑːnɑːbaɪ/)",
    "legionaris": "[legionaris](/lɛɡiːoʊnɑːrɪs/)",
    "oppidum": "[oppidum](/oʊpiːdʊm/)",
    "castellum": "[castellum](/kɑːstɛlʊm/)",
    "castella": "[castella](/kɑːstɛlɑː/)",
    "vici": "[vici](/viːkiː/)",
    "civitates": "[civitates](/kiːviːtɑːtɛs/)",
    "aesculapius": "[Aesculapius](/iːskjuːlɑːpiːʊs/)",
    "chirurgus": "[chirurgus](/kiːrʊrɡʊs/)",
    "ocularius": "[ocularius](/oʊkjuːlɑːriːʊs/)",
    "steatopygous": "[steatopygous](/stiːətəʊpaɪɡəs/)",
}

def expand_roman_numerals(text: str) -> str:
    """Expands Roman numerals for military units and historical rulers."""
    def replace_legion_pair(m):
        prefix, n1, conn, n2 = m.group(1), m.group(2).upper(), m.group(3), m.group(4).upper()
        return f"{prefix} {ROMAN_ORDINALS.get(n1, n1)} {conn} {ROMAN_ORDINALS.get(n2, n2)}"

    text = re.sub(r'\b(legions?)\s+([IVXLCDM]+)\s+(and|&)\s+([IVXLCDM]+)\b', replace_legion_pair, text, flags=re.IGNORECASE)

    def replace_legion(m):
        prefix, num = m.group(1), m.group(2).upper()
        return f"{prefix} {ROMAN_ORDINALS.get(num, num)}"

    text = re.sub(r'\b(legions?)\s+([IVXLCDM]+)\b', replace_legion, text, flags=re.IGNORECASE)

    rulers = (
        r'(?:Gordian|Antoninus|Hadrian|Justinian|Trajan|Severus|'
        r'Merneptah|Ramesses|Constantine|Diocletian|Claudius|Marcus\s+Aurelius)'
    )
    def replace_ruler(m):
        ruler, num = m.group(1), m.group(2).upper()
        return f"{ruler} the {ROMAN_ORDINALS.get(num, num)}"

    text = re.sub(rf'\b({rulers})\s+([IVXLCDM]+)\b', replace_ruler, text, flags=re.IGNORECASE)
    return text

def serbian_to_kokoro_ipa(word: str) -> str:
    """Converts Serbian orthography into Kokoro-compatible phonemes."""
    clean = re.sub(r'[^\w]', '', word).lower()
    pairs = [
        ('dž', 'dʒ'), ('dj', 'dʒ'), ('đ', 'dʒ'), ('lj', 'lj'), ('nj', 'nj'),
        ('č', 'tʃ'), ('ć', 'tʃ'), ('ž', 'ʒ'), ('š', 'ʃ'), ('c', 'ts')
    ]
    for s, ipa in pairs:
        clean = clean.replace(s, ipa)

    clean = re.sub(r'(^|[^aeiou])r([^aeiou]|$)', r'\1ər\2', clean)

    vowels = [('a', 'ɑː'), ('e', 'ɛ'), ('i', 'iː'), ('o', 'oʊ'), ('u', 'uː')]
    for s, ipa in vowels:
        clean = clean.replace(s, ipa)

    return clean

def process_text(text: str) -> str:
    """Normalizes Roman numerals, Latin archeological terms, and Serbian proper nouns."""
    text = expand_roman_numerals(text)

    for term, ipa_override in SPECIAL_TERMS.items():
        text = re.sub(rf'\b{term}\b', ipa_override, text, flags=re.IGNORECASE)

    pattern = r'(\[[^\]]+\]\(/[^/]+/\))|([a-zA-ZčćžšđČĆŽŠĐ]+)'

    def replace_match(match):
        tagged_block = match.group(1)
        if tagged_block:
            return tagged_block
        raw = match.group(2)
        lower = raw.lower()
        if re.search(r'[čćžšđČĆŽŠĐ]', raw) or lower in CUSTOM_SERBIAN_WORDS:
            return f"[{raw}](/{serbian_to_kokoro_ipa(raw)}/)"
        return raw

    return re.sub(pattern, replace_match, text)

# =====================================================================
# 2. AUDIO ENCODING (MP3 96k mono, 24kHz)
# =====================================================================
def save_mp3(audio_data: np.ndarray, output_path: str, sample_rate: int = SAMPLE_RATE, bitrate: str = BITRATE):
    """Pipes raw PCM float32 directly through FFmpeg to produce a 96k mono MP3."""
    if not output_path.lower().endswith(".mp3"):
        output_path = os.path.splitext(output_path)[0] + ".mp3"

    if shutil.which("ffmpeg"):
        pcm_bytes = audio_data.astype(np.float32).tobytes()
        cmd = [
            "ffmpeg", "-y",
            "-f", "f32le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "libmp3lame",
            "-b:a", bitrate,
            "-ac", "1",
            "-ar", str(sample_rate),
            output_path
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = proc.communicate(input=pcm_bytes)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
    else:
        try:
            sf.write(output_path, audio_data, sample_rate, format="MP3")
        except Exception:
            fallback_wav = os.path.splitext(output_path)[0] + ".wav"
            sf.write(fallback_wav, audio_data, sample_rate)
            raise RuntimeError(f"FFmpeg not found. Saved fallback WAV: {fallback_wav}")

# =====================================================================
# 3. TEXT COMPARISON NORMALIZER (from align_demo.py)
# =====================================================================
def normalize_word(w: str) -> str:
    """Normalizes tokens for acoustic-to-manuscript sequence matching."""
    t = w.lower()
    replacements = {'ž': 'z', 'č': 'c', 'ć': 'c', 'š': 's', 'đ': 'dj', 'oe': 'e', 'ae': 'e'}
    for src, dst in replacements.items():
        t = t.replace(src, dst)
    return re.sub(r'[^\w]', '', t)

# =====================================================================
# 4. MASTER SINGLE-STOP BUILD ENGINE
# =====================================================================
def build_stop(
    stop_name: str = "introduction",
    voice: str = "af_bella",
    tts_pipeline: KPipeline = None,
    whisper_model=None,
    align_model_tuple=None
):
    """
    Executes the entire generation pipeline for a single stop:
    1. Reads manuscripts/<stop_name>.txt
    2. Runs phonetic normalization and Kokoro synthesis -> audio_output/<stop_name>.mp3
    3. Runs WhisperX acoustic transcription and global opcode alignment
    4. Writes synchronized cues -> audio_output/<stop_name>_cues.json
    """
    manuscript_path = os.path.join(MANUSCRIPTS_DIR, f"{stop_name}.txt")
    mp3_path = os.path.join(OUTPUT_DIR, f"{stop_name}.mp3")
    cues_path = os.path.join(OUTPUT_DIR, f"{stop_name}_cues.json")

    if not os.path.exists(manuscript_path):
        raise FileNotFoundError(f"Manuscript not found at: {manuscript_path}")

    print(f"\n[{stop_name.upper()}] Reading manuscript: {manuscript_path}")
    with open(manuscript_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    # --- Phase A: Kokoro Synthesis ---
    if tts_pipeline is None:
        print(f"[{stop_name.upper()}] Initializing Kokoro TTS...")
        tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

    print(f"[{stop_name.upper()}] Normalizing text and synthesizing audio ({voice})...")
    normalized_text = process_text(raw_text)

    chunks = []
    for _, _, audio_chunk in tts_pipeline(normalized_text, voice=voice, speed=1.0):
        chunks.append(audio_chunk)

    if not chunks:
        raise RuntimeError(f"Kokoro produced no audio for {stop_name}")

    full_audio = np.concatenate(chunks)
    save_mp3(full_audio, mp3_path, sample_rate=SAMPLE_RATE, bitrate=BITRATE)
    print(f"[{stop_name.upper()}] MP3 saved to: {mp3_path}")

    # --- Phase B: Token & Sentence Preparation ---
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text) if s.strip()]
    ms_words_norm = []
    sentence_word_ranges = []

    for sent in sentences:
        raw_tokens = [w for w in sent.split() if normalize_word(w)]
        start_i = len(ms_words_norm)
        for w in raw_tokens:
            ms_words_norm.append(normalize_word(w))
        end_i = len(ms_words_norm) - 1
        sentence_word_ranges.append((sent, start_i, end_i))

    # --- Phase C: WhisperX Forced Alignment ---
    print(f"[{stop_name.upper()}] Loading audio into WhisperX...")
    audio = whisperx.load_audio(mp3_path)
    audio_duration = round(len(audio) / 16000.0, 2)

    if whisper_model is None:
        print(f"[{stop_name.upper()}] Loading Whisper model (small.en)...")
        whisper_model = whisperx.load_model("small.en", WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    print(f"[{stop_name.upper()}] Transcribing...")
    transcript_result = whisper_model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE)

    if align_model_tuple is None:
        print(f"[{stop_name.upper()}] Loading WhisperX alignment model...")
        align_model, metadata = whisperx.load_align_model(
            language_code=transcript_result["language"],
            device=WHISPER_DEVICE
        )
    else:
        align_model, metadata = align_model_tuple

    print(f"[{stop_name.upper()}] Aligning audio words...")
    aligned_result = whisperx.align(
        transcript_result["segments"],
        align_model,
        metadata,
        audio,
        WHISPER_DEVICE,
        return_char_alignments=False
    )

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

    # --- Phase D: Global Opcode Sequence Matching ---
    matcher = difflib.SequenceMatcher(None, ms_words_norm, audio_words_norm, autojunk=False)
    ms_word_start = [0.0] * len(ms_words_norm)
    ms_word_end = [0.0] * len(ms_words_norm)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ms_word_start[i1 + k] = audio_words[j1 + k]["start"]
                ms_word_end[i1 + k] = audio_words[j1 + k]["end"]
        elif tag == "replace":
            span_start = audio_words[j1]["start"]
            span_end = audio_words[j2 - 1]["end"]
            count = i2 - i1
            dur = (span_end - span_start) / max(count, 1)
            for k in range(count):
                ms_word_start[i1 + k] = span_start + k * dur
                ms_word_end[i1 + k] = span_start + (k + 1) * dur
        elif tag == "delete":
            fallback_t = audio_words[j1]["start"] if j1 < len(audio_words) else (audio_words[-1]["end"] if audio_words else 0.0)
            for i in range(i1, i2):
                ms_word_start[i] = fallback_t
                ms_word_end[i] = fallback_t

    # --- Phase E: Sentence Cue Building & Fallback Clamping ---
    cues = []
    for sent_text, start_i, end_i in sentence_word_ranges:
        cues.append({
            "start": round(ms_word_start[start_i], 2),
            "end": round(ms_word_end[end_i], 2),
            "text": sent_text
        })

    if cues and cues[-1]["end"] < cues[-1]["start"]:
        cues[-1]["end"] = audio_duration

    with open(cues_path, "w", encoding="utf-8") as f:
        json.dump(cues, f, ensure_ascii=False, indent=2)

    print(f"[{stop_name.upper()}] Build complete! Exported {len(cues)} sentence cues to: {cues_path}")
    return mp3_path, cues_path

# =====================================================================
# 5. CLI INTERFACE
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless single-stop build engine for museum audio guides.")
    parser.add_argument("--stop", type=str, default="introduction", help="Stop name matching manuscript filename (without .txt)")
    parser.add_argument("--voice", type=str, default="af_bella", help="Kokoro voice model identifier")
    args = parser.parse_args()

    build_stop(stop_name=args.stop, voice=args.voice)