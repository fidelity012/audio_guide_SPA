"""
build_test_v2.py - Real TTS + WhisperX + Visual Cueing Test Builder
1. Reads manuscripts/<stop>.txt & lexicon/lexicon.json
2. Synthesizes 96k mono MP3 via Kokoro-82M & FFmpeg
3. Runs WhisperX forced alignment + difflib global opcode mapping
4. Scales visual chapters (stop.images) to actual audio duration
5. Generates test_player_v2.html wired to the real local MP3
"""

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
import torch
from kokoro import KPipeline
import whisperx

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANUSCRIPTS_DIR = os.path.join(BASE_DIR, "manuscripts")
OUTPUT_DIR = os.path.join(BASE_DIR, "audio_output")
DEFAULT_LEXICON_PATH = os.path.join(BASE_DIR, "lexicon", "lexicon.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MANUSCRIPTS_DIR, exist_ok=True)

SAMPLE_RATE = 24000
BITRATE = "96k"

# Hardware auto-detection (CUDA GPU vs CPU)
if torch.cuda.is_available():
    WHISPER_DEVICE = "cuda"
    WHISPER_COMPUTE_TYPE = "float16"
    WHISPER_BATCH_SIZE = 16
else:
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"
    WHISPER_BATCH_SIZE = 4

DEFAULT_TEST_MANUSCRIPT = """Viminacium, a major Roman city and military camp on the Danube, had a sophisticated medical community.
The medicus-chirurgus was buried with a rich set of surgical tools, including tweezers, probes, and a bronze instrumentarium box.
Excavated graves in the southern necropolis have yielded various specialised instruments, including scalpels, bone forceps, and ocular needles.
These finds testify to the complex surgical procedures performed here between the 2nd and 4th centuries."""

DEFAULT_TEST_IMAGES = [
    {
        "ratio_start": 0.0,
        "ratio_end": 0.30,
        "src": "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?w=600&auto=format&fit=crop&q=80",
        "caption": "Showcase 4: Roman Viminacium Medical Collection"
    },
    {
        "ratio_start": 0.30,
        "ratio_end": 0.65,
        "src": "https://images.unsplash.com/photo-1584362917165-526a968579e8?w=600&auto=format&fit=crop&q=80",
        "caption": "Bronze Instrumentarium Box & Surgical Probes"
    },
    {
        "ratio_start": 0.65,
        "ratio_end": 1.0,
        "src": "https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?w=600&auto=format&fit=crop&q=80",
        "caption": "Specialized Ocular & Cataract Needles"
    }
]

# =====================================================================
# 1. LEXICON & TEXT NORMALIZATION
# =====================================================================
ROMAN_ORDINALS = {
    'I': 'First', 'II': 'Second', 'III': 'Third', 'IV': 'Fourth',
    'V': 'Fifth', 'VI': 'Sixth', 'VII': 'Seventh', 'VIII': 'Eighth',
    'IX': 'Ninth', 'X': 'Tenth', 'XI': 'Eleventh', 'XII': 'Twelfth',
    'XIII': 'Thirteenth', 'XIV': 'Fourteenth', 'XV': 'Fifteenth',
    'XVI': 'Sixteenth', 'XVII': 'Seventeenth', 'XVIII': 'Eighteenth',
    'XIX': 'Nineteenth', 'XX': 'Twentieth'
}

def load_lexicon(path: str = None) -> dict:
    path = path or DEFAULT_LEXICON_PATH
    lex = {"serbian_words": set(), "diacritic_restore": {}, "ipa": {}}
    if not os.path.exists(path):
        print(f"[LEXICON] Not found at {path}; using built-in phonetic rules.")
        return lex
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lex["serbian_words"] = {w.lower() for w in data.get("serbian_words", [])}
    lex["diacritic_restore"] = {k.lower(): v for k, v in data.get("diacritic_restore", {}).items()}
    lex["ipa"] = {k.lower(): v for k, v in data.get("ipa", {}).items()}
    print(f"[LEXICON] Loaded {path}")
    return lex

def expand_roman_numerals(text: str) -> str:
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

def process_text(text: str, lexicon: dict) -> str:
    text = expand_roman_numerals(text)

    # Apply explicit IPA overrides from lexicon.json
    for term, ipa_val in lexicon.get("ipa", {}).items():
        clean_ipa = ipa_val.strip("/")
        text = re.sub(
            rf'\b({re.escape(term)})\b',
            lambda m: f"[{m.group(1)}](/{clean_ipa}/)",
            text,
            flags=re.IGNORECASE
        )

    pattern = r'(\[[^\]]+\]\(/[^/]+/\))|([a-zA-ZčćžšđČĆŽŠĐ]+)'

    def replace_match(match):
        if match.group(1):
            return match.group(1)
        raw = match.group(2)
        lower = raw.lower()
        restored = lexicon.get("diacritic_restore", {}).get(lower, raw)
        if re.search(r'[čćžšđČĆŽŠĐ]', restored) or lower in lexicon.get("serbian_words", set()):
            return f"[{raw}](/{serbian_to_kokoro_ipa(restored)}/)"
        return raw

    return re.sub(pattern, replace_match, text)

# =====================================================================
# 2. AUDIO ENCODING & WORD NORMALIZATION
# =====================================================================
def save_mp3(audio_data: np.ndarray, output_path: str, sample_rate: int = SAMPLE_RATE, bitrate: str = BITRATE):
    if shutil.which("ffmpeg"):
        pcm_bytes = audio_data.astype(np.float32).tobytes()
        cmd = [
            "ffmpeg", "-y", "-f", "f32le", "-ar", str(sample_rate), "-ac", "1",
            "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", bitrate,
            "-ac", "1", "-ar", str(sample_rate), output_path
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = proc.communicate(input=pcm_bytes)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
    else:
        sf.write(output_path, audio_data, sample_rate, format="MP3")

def normalize_word(w: str) -> str:
    t = w.lower()
    replacements = {'ž': 'z', 'č': 'c', 'ć': 'c', 'š': 's', 'đ': 'dj', 'oe': 'e', 'ae': 'e'}
    for src, dst in replacements.items():
        t = t.replace(src, dst)
    return re.sub(r'[^\w]', '', t)

# =====================================================================
# 3. HTML TEST HARNESS GENERATOR
# =====================================================================
def generate_test_player_html(stop_data, html_path):
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>{stop_data.get('title', 'Audio Guide v2 Test')}</title>
  <style>
    :root {{
      --primary-accent: #C65D3B;
      --bg-cream: #F5F2ED;
      --card-bg: #FFFFFF;
      --text-dark: #2E2E2E;
      --text-muted: #7A7570;
      --border-subtle: #E2DDD6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: #222;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      padding: 10px;
    }}
    .phone-simulator {{
      width: 100%;
      max-width: 400px;
      height: 840px;
      max-height: 94vh;
      background-color: var(--bg-cream);
      border-radius: 36px;
      overflow: hidden;
      box-shadow: 0 20px 40px rgba(0,0,0,0.4);
      display: flex;
      flex-direction: column;
      position: relative;
      border: 8px solid #1a1a1a;
    }}
    .screen-content {{
      padding: 24px 20px 16px;
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
    }}
    .header-title {{ text-align: center; margin-bottom: 14px; }}
    .header-title h1 {{
      font-size: 1.35rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      color: var(--text-dark);
      text-transform: uppercase;
    }}
    .slide-media-stage {{
      position: relative;
      width: 100%;
      height: 13rem;
      border-radius: 12px;
      overflow: hidden;
      background-color: #1e1e1e;
      margin-bottom: 14px;
      transition: height 0.35s cubic-bezier(0.4, 0, 0.2, 1);
      flex-shrink: 0;
      box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }}
    .phone-simulator.text-expanded .slide-media-stage {{
      height: 6.8rem;
      margin-bottom: 8px;
    }}
    .stage-img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: opacity 0.25s ease-in-out;
    }}
    .stage-caption {{
      position: absolute;
      bottom: 0; left: 0; right: 0;
      background: linear-gradient(transparent, rgba(0,0,0,0.75));
      color: #FFF;
      font-size: 0.75rem;
      padding: 8px 10px 6px;
      text-align: center;
      pointer-events: none;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .stage-nav {{
      position: absolute;
      top: 50%;
      transform: translateY(-50%);
      width: 32px; height: 32px;
      background: rgba(0,0,0,0.45);
      border: 1px solid rgba(255,255,255,0.3);
      color: #fff;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.2rem;
      z-index: 5;
    }}
    .stage-nav:hover {{ background: rgba(0,0,0,0.75); }}
    .stage-nav.prev {{ left: 8px; }}
    .stage-nav.next {{ right: 8px; }}
    .audio-control-cluster {{
      display: flex;
      flex-direction: column;
      align-items: center;
      margin-bottom: 12px;
      flex-shrink: 0;
    }}
    .btn-row {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 24px;
      margin-bottom: 8px;
    }}
    .btn-skip {{
      background: none;
      border: 2px solid var(--primary-accent);
      color: var(--primary-accent);
      width: 44px; height: 44px;
      border-radius: 50%;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .btn-skip:active {{ background: var(--primary-accent); color: #FFF; }}
    .btn-play-pause {{
      background: var(--primary-accent);
      border: none;
      color: #FFF;
      width: 58px; height: 58px;
      border-radius: 50%;
      font-size: 1.5rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 10px rgba(198, 93, 59, 0.35);
    }}
    .timeline-row {{
      width: 100%;
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-muted);
    }}
    .time-scrubber {{
      flex: 1;
      accent-color: var(--primary-accent);
      height: 4px;
      cursor: pointer;
    }}
    .toggle-transcript {{
      width: 100%;
      padding: 10px 14px;
      background: #FFF;
      border: 2px solid var(--border-subtle);
      border-radius: 10px;
      font-size: 0.9rem;
      font-weight: 800;
      letter-spacing: 0.05em;
      color: var(--text-dark);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      margin-bottom: 8px;
      flex-shrink: 0;
    }}
    .transcript-drawer {{
      display: none;
      flex: 1;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      background: var(--card-bg);
      border-radius: 12px;
      padding: 14px 16px;
      border: 1px solid var(--border-subtle);
      line-height: 1.6;
      font-size: 0.95rem;
      color: #333;
    }}
    .phone-simulator.text-expanded .transcript-drawer {{ display: block; }}
    .cue {{
      transition: background-color 0.2s ease;
      padding: 2px 4px;
      border-radius: 4px;
      cursor: pointer;
    }}
    .cue.active {{
      background-color: rgba(198, 93, 59, 0.16);
      box-shadow: -3px 0 0 var(--primary-accent);
      font-weight: 600;
      color: #111;
    }}
  </style>
</head>
<body>
<div class="phone-simulator" id="simulator">
  <div class="screen-content">
    <div class="header-title"><h1>{stop_data.get('title', 'ROMAN MEDICINE')}</h1></div>
    <div class="slide-media-stage">
      <button class="stage-nav prev" id="prev-artifact" aria-label="Previous artifact">&lsaquo;</button>
      <img class="stage-img" id="stage-img" src="{stop_data['images'][0]['src']}" alt="Artifact">
      <button class="stage-nav next" id="next-artifact" aria-label="Next artifact">&rsaquo;</button>
      <span class="stage-caption" id="stage-caption">{stop_data['images'][0].get('caption', '')}</span>
    </div>
    <div class="audio-control-cluster">
      <div class="btn-row">
        <button class="btn-skip" id="btn-back">&#8634; 10</button>
        <button class="btn-play-pause" id="btn-play">&#9658;</button>
        <button class="btn-skip" id="btn-fwd">10 &#8635;</button>
      </div>
      <div class="timeline-row">
        <span id="time-current">0:00</span>
        <input type="range" class="time-scrubber" id="scrubber" min="0" value="0" step="0.1">
        <span id="time-duration">0:00</span>
      </div>
    </div>
    <audio id="audio-el" src="{stop_data['audio']}" preload="metadata"></audio>
    <button class="toggle-transcript" id="toggle-btn" aria-expanded="false">
      <span id="toggle-label">REVEAL TEXT</span>
      <span class="arrow-icon" id="toggle-arrow">&#9660;</span>
    </button>
    <div class="transcript-drawer" id="drawer">
      <div id="transcript-body"></div>
    </div>
  </div>
</div>
<script>
  const STOP_DATA = {json.dumps(stop_data, ensure_ascii=False, indent=2)};
  const simulator = document.getElementById('simulator');
  const audio = document.getElementById('audio-el');
  const playBtn = document.getElementById('btn-play');
  const backBtn = document.getElementById('btn-back');
  const fwdBtn = document.getElementById('btn-fwd');
  const scrubber = document.getElementById('scrubber');
  const timeCur = document.getElementById('time-current');
  const timeDur = document.getElementById('time-duration');
  const toggleBtn = document.getElementById('toggle-btn');
  const toggleLabel = document.getElementById('toggle-label');
  const toggleArrow = document.getElementById('toggle-arrow');
  const transcriptBody = document.getElementById('transcript-body');
  const stageImg = document.getElementById('stage-img');
  const stageCaption = document.getElementById('stage-caption');
  const nextArtifactBtn = document.getElementById('next-artifact');
  const prevArtifactBtn = document.getElementById('prev-artifact');

  function formatTime(sec) {{
    if (isNaN(sec)) return "0:00";
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ":" + (s < 10 ? "0" : "") + s;
  }}

  STOP_DATA.cues.forEach((cue, i) => {{
    const span = document.createElement('span');
    span.className = 'cue';
    span.dataset.start = cue.start;
    span.dataset.end = cue.end;
    span.textContent = cue.text + ' ';
    transcriptBody.appendChild(span);
  }});

  playBtn.addEventListener('click', () => {{
    if (audio.paused) audio.play();
    else audio.pause();
  }});
  audio.addEventListener('play', () => {{ playBtn.innerHTML = '&#10074;&#10074;'; }});
  audio.addEventListener('pause', () => {{ playBtn.innerHTML = '&#9658;'; }});

  backBtn.addEventListener('click', () => {{ audio.currentTime = Math.max(0, audio.currentTime - 10); }});
  fwdBtn.addEventListener('click', () => {{ audio.currentTime = Math.min(audio.duration || 9999, audio.currentTime + 10); }});

  audio.addEventListener('loadedmetadata', () => {{
    scrubber.max = audio.duration || 100;
    timeDur.textContent = formatTime(audio.duration);
  }});

  let isScrubbing = false;
  scrubber.addEventListener('input', () => {{
    isScrubbing = true;
    timeCur.textContent = formatTime(scrubber.value);
  }});
  scrubber.addEventListener('change', () => {{
    audio.currentTime = parseFloat(scrubber.value);
    isScrubbing = false;
  }});

  audio.addEventListener('timeupdate', () => {{
    const t = audio.currentTime;
    if (!isScrubbing) {{
      scrubber.value = t;
      timeCur.textContent = formatTime(t);
    }}
    if (STOP_DATA.images && STOP_DATA.images.length > 0) {{
      const activeImg = STOP_DATA.images.find(img => t >= img.start && t <= img.end);
      if (activeImg && stageImg.dataset.currentSrc !== activeImg.src) {{
        stageImg.style.opacity = '0.25';
        setTimeout(() => {{
          stageImg.src = activeImg.src;
          stageImg.dataset.currentSrc = activeImg.src;
          stageImg.style.opacity = '1';
          stageCaption.textContent = activeImg.caption || '';
        }}, 140);
      }}
    }}
    transcriptBody.querySelectorAll('.cue').forEach(cue => {{
      const start = parseFloat(cue.dataset.start);
      const end = parseFloat(cue.dataset.end);
      cue.classList.toggle('active', t >= start && t <= end);
    }});
  }});

  nextArtifactBtn.addEventListener('click', () => {{
    const next = STOP_DATA.images.find(img => img.start > audio.currentTime + 0.5);
    if (next) {{ audio.currentTime = next.start; audio.play(); }}
  }});

  prevArtifactBtn.addEventListener('click', () => {{
    const t = audio.currentTime;
    const curIdx = STOP_DATA.images.findIndex((img, i) => {{
      const next = STOP_DATA.images[i + 1];
      return t >= img.start && (!next || t < next.start);
    }});
    if (curIdx > 0 && (t - STOP_DATA.images[curIdx].start < 3.0)) {{
      audio.currentTime = STOP_DATA.images[curIdx - 1].start;
    }} else if (curIdx >= 0) {{
      audio.currentTime = STOP_DATA.images[curIdx].start;
    }}
    audio.play();
  }});

  toggleBtn.addEventListener('click', () => {{
    const isExpanded = simulator.classList.toggle('text-expanded');
    toggleBtn.setAttribute('aria-expanded', isExpanded);
    toggleLabel.textContent = isExpanded ? 'HIDE TEXT' : 'REVEAL TEXT';
    toggleArrow.innerHTML = isExpanded ? '&#9650;' : '&#9660;';
    if (isExpanded) {{
      const activeCue = transcriptBody.querySelector('.cue.active');
      if (activeCue) activeCue.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
    }}
  }});

  transcriptBody.addEventListener('click', (e) => {{
    const cue = e.target.closest('.cue');
    if (cue && cue.dataset.start !== undefined) {{
      audio.currentTime = parseFloat(cue.dataset.start);
      audio.play();
    }}
  }});
</script>
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

# =====================================================================
# 4. MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Build & Test v2 Stop with real Kokoro TTS + WhisperX")
    parser.add_argument("--stop", type=str, default="test_medicine", help="Manuscript filename (without .txt)")
    parser.add_argument("--voice", type=str, default="af_bella", help="Kokoro voice preset")
    args = parser.parse_args()

    stop_id = args.stop
    manuscript_file = os.path.join(MANUSCRIPTS_DIR, f"{stop_id}.txt")
    mp3_filename = f"{stop_id}_v2.mp3"
    mp3_path = os.path.join(OUTPUT_DIR, mp3_filename)
    output_json = os.path.join(OUTPUT_DIR, f"{stop_id}_stop_v2.json")
    output_html = os.path.join(BASE_DIR, "test_player_v2.html")

    if not os.path.exists(manuscript_file):
        with open(manuscript_file, "w", encoding="utf-8") as f:
            f.write(DEFAULT_TEST_MANUSCRIPT)
        print(f"[MANUSCRIPT] Created default manuscript: {manuscript_file}")

    with open(manuscript_file, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    lexicon = load_lexicon()
    normalized_text = process_text(raw_text, lexicon)

    # 1. Kokoro TTS Synthesis
    print(f"[{stop_id.upper()}] Synthesizing narration via Kokoro-82M ({args.voice})...")
    tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    chunks = [chunk for _, _, chunk in tts_pipeline(normalized_text, voice=args.voice, speed=1.0)]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio.")

    full_audio = np.concatenate(chunks)
    save_mp3(full_audio, mp3_path, sample_rate=SAMPLE_RATE, bitrate=BITRATE)
    print(f"[{stop_id.upper()}] Saved MP3: {mp3_path}")

    # 2. WhisperX Forced Alignment + Global Opcode Matching
    print(f"[{stop_id.upper()}] Running WhisperX alignment on {WHISPER_DEVICE}...")
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

    audio = whisperx.load_audio(mp3_path)
    audio_duration = round(len(audio) / 16000.0, 2)

    whisper_model = whisperx.load_model("small.en", WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    result = whisper_model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE)

    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=WHISPER_DEVICE)
    aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, WHISPER_DEVICE, return_char_alignments=False)

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

    cues = []
    for sent_text, start_i, end_i in sentence_word_ranges:
        cues.append({
            "start": round(ms_word_start[start_i], 2),
            "end": round(ms_word_end[end_i], 2),
            "text": sent_text
        })
    if cues and cues[-1]["end"] < cues[-1]["start"]:
        cues[-1]["end"] = audio_duration

    # 3. Map image chapters proportionally to actual synthesized audio duration
    images = []
    for img in DEFAULT_TEST_IMAGES:
        images.append({
            "start": round(img["ratio_start"] * audio_duration, 2),
            "end": round(img["ratio_end"] * audio_duration, 2),
            "src": img["src"],
            "caption": img["caption"]
        })

    stop_data = {
        "id": stop_id,
        "room": "viminacium",
        "title": stop_id.replace("_", " ").title(),
        "audio": f"audio_output/{mp3_filename}",
        "images": images,
        "cues": cues
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stop_data, f, ensure_ascii=False, indent=2)

    generate_test_player_html(stop_data, output_html)
    print(f"\n[DONE] Generated real narration ({audio_duration}s) -> http://localhost:8000/test_player_v2.html")

if __name__ == "__main__":
    main()