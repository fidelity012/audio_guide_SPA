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
DEFAULT_LEXICON_PATH = os.path.join(BASE_DIR, "lexicon", "lexicon.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 24000
BITRATE = "96k"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_BATCH_SIZE = 4

WORD_RE = r"[A-Za-zČĆŽŠĐčćžšđ]+"

# =====================================================================
# 1. PER-MUSEUM LEXICON (lexicon.json)
# =====================================================================
def load_lexicon(path: str = None) -> dict:
    """
    Loads the museum-specific pronunciation lexicon. Keys are lowercase.
      serbian_words      : names converted with the rule-based Serbian -> IPA converter
      diacritic_restore  : ASCII spelling -> true spelling (pronunciation only)
      ipa                : hand-written IPA overrides (win over everything else)
      roman_exclude      : extra words after which a Roman numeral is NOT expanded
    A missing file is fine: the generic rules still run.
    """
    path = path or DEFAULT_LEXICON_PATH
    lex = {"serbian_words": set(), "diacritic_restore": {}, "ipa": {}, "roman_exclude": set()}
    if not os.path.exists(path):
        print(f"[LEXICON] Not found at {path}; using generic rules only.")
        return lex
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lex["serbian_words"] = {w.lower() for w in data.get("serbian_words", [])}
    lex["diacritic_restore"] = {k.lower(): v for k, v in data.get("diacritic_restore", {}).items()}
    lex["ipa"] = {k.lower(): v for k, v in data.get("ipa", {}).items()}
    lex["roman_exclude"] = {w.lower() for w in data.get("roman_exclude", [])}
    print(f"[LEXICON] Loaded {path}: {len(lex['serbian_words'])} Serbian, "
          f"{len(lex['ipa'])} IPA, {len(lex['diacritic_restore'])} restore entries.")
    return lex

# =====================================================================
# 2. TEXT NORMALIZATION FOR TTS (display text is never touched)
# =====================================================================
ROMAN_ORDINALS = {
    'I': 'First', 'II': 'Second', 'III': 'Third', 'IV': 'Fourth',
    'V': 'Fifth', 'VI': 'Sixth', 'VII': 'Seventh', 'VIII': 'Eighth',
    'IX': 'Ninth', 'X': 'Tenth', 'XI': 'Eleventh', 'XII': 'Twelfth',
    'XIII': 'Thirteenth', 'XIV': 'Fourteenth', 'XV': 'Fifteenth',
    'XVI': 'Sixteenth', 'XVII': 'Seventeenth', 'XVIII': 'Eighteenth',
    'XIX': 'Nineteenth', 'XX': 'Twentieth'
}

ROMAN_CARDINALS = {
    'I': 'One', 'II': 'Two', 'III': 'Three', 'IV': 'Four', 'V': 'Five',
    'VI': 'Six', 'VII': 'Seven', 'VIII': 'Eight', 'IX': 'Nine', 'X': 'Ten',
    'XI': 'Eleven', 'XII': 'Twelve', 'XIII': 'Thirteen', 'XIV': 'Fourteen',
    'XV': 'Fifteen', 'XVI': 'Sixteen', 'XVII': 'Seventeen', 'XVIII': 'Eighteen',
    'XIX': 'Nineteen', 'XX': 'Twenty'
}

# Valid Roman numeral (uppercase only, non-empty)
ROMAN = r'(?=[IVXLCDM])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})'

# Words after which a numeral is a label, not a regnal number ("World War II")
DEFAULT_ROMAN_EXCLUDE = {
    "war", "world", "chapter", "part", "phase", "level", "act", "table", "figure", "fig",
    "vol", "volume", "section", "type", "group", "class", "stage", "period", "room",
    "gallery", "hall", "size", "grade", "tier", "page", "plate", "wave", "operation",
    "legion", "legions",
}

def expand_roman_numerals(text: str, extra_exclude=frozenset()) -> str:
    """
    Expands Roman numerals, generically:
      - In any sentence mentioning a legion, a numeral before a capitalised name
        ("IV Flavia", "and V Macedonica") or right after "legion(s)" becomes an ordinal.
      - Capitalised name + numeral of 2+ letters ("Gordian III") -> "Gordian the Third".
    """
    exclude = DEFAULT_ROMAN_EXCLUDE | {w.lower() for w in extra_exclude}

    def ordinal(num):
        return ROMAN_ORDINALS.get(num, num)

    def regnal(m):
        prev, num = m.group(1), m.group(2)
        if prev.lower() in exclude or num not in ROMAN_ORDINALS or len(num) < 2:
            return m.group(0)
        return f"{prev} the {ROMAN_ORDINALS[num]}"

    pieces = re.split(r'(?<=[.!?])(\s+)', text)   # keeps the whitespace separators
    out = []
    for piece in pieces:
        if re.search(r'\b[Ll]egions?\b', piece):
            # After the word "legion(s)": cardinal ("Legion Seven Claudia").
            # Before it ("the VII Claudia and IV Flavia legions"): ordinal ("Seventh").
            def legion_num(m, whole=piece):
                num = m.group(1)
                after_legion = re.search(r'\b[Ll]egions?\b', whole[:m.start()])
                return (ROMAN_CARDINALS if after_legion else ROMAN_ORDINALS).get(num, num)
            piece = re.sub(rf'\b({ROMAN})\b(?=\s+[A-Z])', legion_num, piece)
            # "Legion IV and XIII", "legions IV, V and VI": every numeral in the list
            chain = rf'(\b[Ll]egions?\s+)({ROMAN}(?:\s*(?:,|and|&)\s*(?:and\s+)?{ROMAN})*)\b'
            piece = re.sub(chain,
                           lambda m: m.group(1) + re.sub(rf'\b({ROMAN})\b',
                                                         lambda n: ROMAN_CARDINALS.get(n.group(1), n.group(1)),
                                                         m.group(2)),
                           piece)
        piece = re.sub(rf'\b([A-Z][a-z]+)\s+({ROMAN})\b', regnal, piece)
        out.append(piece)
    return ''.join(out)

def normalize_numbers_and_units(text: str) -> str:
    """Glued eras, numeric ranges, 'c.' and common units, for speech only."""
    text = re.sub(r'(?<=\d)(AD|BC|BCE|CE)\b', r' \1', text)          # 117AD -> 117 AD
    text = re.sub(r'(?<=\d)\s*[–—-]\s*(?=\d)', ' to ', text)          # 4500–2200 -> 4500 to 2200
    text = re.sub(r'\bc\.\s*(?=\d)', 'circa ', text)                  # c. 300 -> circa 300
    text = re.sub(r'(?<=\d)\s*km\b', ' kilometers', text)
    text = re.sub(r'(?<=\d)\s*m\b(?!\w)', ' meters', text)
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

    # Kokoro's vocab has no ASCII 'g'; it needs IPA 'ɡ' (U+0261), else the sound is dropped.
    clean = clean.replace('g', 'ɡ')

    clean = re.sub(r'(^|[^aeiou])r([^aeiou]|$)', r'\1ər\2', clean)

    vowels = [('a', 'ɑː'), ('e', 'ɛ'), ('i', 'iː'), ('o', 'oʊ'), ('u', 'uː')]
    for s, ipa in vowels:
        clean = clean.replace(s, ipa)

    return clean

def process_text(text: str, lex: dict) -> str:
    """Builds the TTS-only text: numerals, dates, units, then lexicon pronunciation tags."""
    text = expand_roman_numerals(text, lex["roman_exclude"])
    text = normalize_numbers_and_units(text)

    pattern = rf'(\[[^\]]+\]\(/[^/]+/\))|({WORD_RE})'

    def replace_match(match):
        tagged_block = match.group(1)
        if tagged_block:
            return tagged_block
        raw = match.group(2)
        low = raw.lower()
        if low in lex["ipa"]:
            return f"[{raw}](/{lex['ipa'][low]}/)"
        spelling = lex["diacritic_restore"].get(low)
        if spelling or low in lex["serbian_words"] or re.search(r'[čćžšđČĆŽŠĐ]', raw):
            return f"[{raw}](/{serbian_to_kokoro_ipa(spelling or raw)}/)"
        return raw

    return re.sub(pattern, replace_match, text)

# =====================================================================
# 3. SENTENCE SPLITTING (abbreviation- and quote-aware)
# =====================================================================
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "st", "no", "nos", "vs", "c", "ca", "cf", "fig", "figs",
    "vol", "approx", "ed", "eds", "pp", "p", "jr", "sr", "e.g", "i.e", "viz", "ca",
}

def split_sentences(text: str) -> list:
    """
    Splits on blank lines (so headings stay separate) and on . ! ? followed by an
    uppercase letter, digit or opening quote. Does not split after abbreviations,
    single initials, or when the next word is lowercase ("in 1952. by Professor").
    Closing quotes/brackets after the period stay with the sentence.
    """
    sentences = []
    for para in re.split(r'\n\s*\n', text):
        para = para.strip()
        if not para:
            continue
        start = 0
        for m in re.finditer(r'[.!?]+["”’\')\]]*\s+', para):
            nxt = para[m.end():m.end() + 1]
            if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in '"“‘\'(['):
                continue
            tail = para[start:m.start()].split()
            last = tail[-1] if tail else ""
            low = last.lower().strip('("“‘\'[')
            if m.group(0).startswith('.') and not m.group(0).startswith('..'):
                if low in ABBREVIATIONS or (len(low) == 1 and last.strip('("“‘\'[').isupper()):
                    continue
            sentences.append(para[start:m.end()].strip())
            start = m.end()
        rest = para[start:].strip()
        if rest:
            sentences.append(rest)
    return sentences

# =====================================================================
# 4. UNKNOWN-WORD REPORT
# =====================================================================
def find_unknown_words(raw_text: str, lex: dict, tts_pipeline) -> dict:
    """
    Words Kokoro's English dictionary doesn't know and the lexicon doesn't cover.
    These are the ones likely to be mispronounced. Falls back to 'capitalised
    mid-sentence words' if the dictionary can't be reached.
    """
    covered = set(lex["ipa"]) | set(lex["serbian_words"]) | set(lex["diacritic_restore"])
    known = None
    try:
        g2p_lex = tts_pipeline.g2p.lexicon
        dicts = (g2p_lex.golds, g2p_lex.silvers)
        known = lambda w: any((w in d) or (w.lower() in d) for d in dicts)
    except Exception:
        known = None

    found = {}
    for sent in split_sentences(raw_text):
        for i, w in enumerate(re.findall(WORD_RE, sent)):
            low = w.lower()
            if len(w) < 3 or w.isupper() or low in covered or re.search(r'[čćžšđČĆŽŠĐ]', w):
                continue
            unknown = (not known(w)) if known else (w[0].isupper() and i > 0)
            if unknown:
                entry = found.setdefault(low, {"word": w, "count": 0, "context": sent[:80]})
                entry["count"] += 1
    return dict(sorted(found.items()))

# =====================================================================
# 5. AUDIO ENCODING (MP3 96k mono, 24kHz)
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
# 6. TEXT COMPARISON NORMALIZER (from align_demo.py)
# =====================================================================
def normalize_word(w: str) -> str:
    """Normalizes tokens for acoustic-to-manuscript sequence matching."""
    t = w.lower()
    replacements = {'ž': 'z', 'č': 'c', 'ć': 'c', 'š': 's', 'đ': 'dj', 'oe': 'e', 'ae': 'e'}
    for src, dst in replacements.items():
        t = t.replace(src, dst)
    return re.sub(r'[^\w]', '', t)

# =====================================================================
# 7. MASTER SINGLE-STOP BUILD ENGINE
# =====================================================================
def build_stop(
    stop_name: str = "introduction",
    voice: str = "af_bella",
    tts_pipeline: KPipeline = None,
    whisper_model=None,
    align_model_tuple=None,
    lexicon: dict = None,
    check_only: bool = False
):
    """
    Executes the entire generation pipeline for a single stop:
    1. Reads manuscripts/<stop_name>.txt
    2. Reports words the TTS is likely to mispronounce (audio_output/<stop_name>_review.txt)
    3. Runs phonetic normalization and Kokoro synthesis -> audio_output/<stop_name>.mp3
    4. Runs WhisperX acoustic transcription and global opcode alignment
    5. Writes synchronized cues -> audio_output/<stop_name>_cues.json
    With check_only=True, stops after step 2 (writes <stop_name>_tts_text.txt for inspection).
    """
    manuscript_path = os.path.join(MANUSCRIPTS_DIR, f"{stop_name}.txt")
    mp3_path = os.path.join(OUTPUT_DIR, f"{stop_name}.mp3")
    cues_path = os.path.join(OUTPUT_DIR, f"{stop_name}_cues.json")
    review_path = os.path.join(OUTPUT_DIR, f"{stop_name}_review.txt")
    tts_text_path = os.path.join(OUTPUT_DIR, f"{stop_name}_tts_text.txt")

    if not os.path.exists(manuscript_path):
        raise FileNotFoundError(f"Manuscript not found at: {manuscript_path}")

    if lexicon is None:
        lexicon = load_lexicon()

    print(f"\n[{stop_name.upper()}] Reading manuscript: {manuscript_path}")
    with open(manuscript_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    # --- Phase A: Kokoro Synthesis ---
    if tts_pipeline is None:
        print(f"[{stop_name.upper()}] Initializing Kokoro TTS...")
        tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

    unknown = find_unknown_words(raw_text, lexicon, tts_pipeline)
    with open(review_path, "w", encoding="utf-8") as f:
        for e in unknown.values():
            f.write(f"{e['word']}\t{e['count']}\t{e['context']}\n")
    if unknown:
        words = ", ".join(e["word"] for e in unknown.values())
        print(f"[{stop_name.upper()}] REVIEW {len(unknown)} word(s) not in the TTS dictionary or lexicon: {words}")
        print(f"[{stop_name.upper()}] Listen to them; fix bad ones in lexicon.json. Details: {review_path}")
    else:
        print(f"[{stop_name.upper()}] No unknown words.")

    normalized_text = process_text(raw_text, lexicon)

    if check_only:
        with open(tts_text_path, "w", encoding="utf-8") as f:
            f.write(normalized_text)
        print(f"[{stop_name.upper()}] Check only. TTS text written to: {tts_text_path}")
        return None, None

    print(f"[{stop_name.upper()}] Synthesizing audio ({voice})...")
    chunks = []
    for _, _, audio_chunk in tts_pipeline(normalized_text, voice=voice, speed=1.0):
        chunks.append(audio_chunk)

    if not chunks:
        raise RuntimeError(f"Kokoro produced no audio for {stop_name}")

    full_audio = np.concatenate(chunks)
    save_mp3(full_audio, mp3_path, sample_rate=SAMPLE_RATE, bitrate=BITRATE)
    print(f"[{stop_name.upper()}] MP3 saved to: {mp3_path}")

    # --- Phase B: Token & Sentence Preparation ---
    sentences = split_sentences(raw_text)
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
# 8. CLI INTERFACE
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless single-stop build engine for museum audio guides.")
    parser.add_argument("--stop", type=str, default="introduction", help="Stop name matching manuscript filename (without .txt)")
    parser.add_argument("--voice", type=str, default="af_bella", help="Kokoro voice model identifier")
    parser.add_argument("--lexicon", type=str, default=None, help="Path to lexicon.json (default: next to this script)")
    parser.add_argument("--check", action="store_true", help="Only report unknown words and write the TTS text; no audio")
    args = parser.parse_args()

    build_stop(stop_name=args.stop, voice=args.voice,
               lexicon=load_lexicon(args.lexicon), check_only=args.check)