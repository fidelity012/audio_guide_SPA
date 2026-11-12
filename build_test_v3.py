"""
build_test_v3.py - End-to-End Test Builder for Museum Audio Guide v3
Features:
1. Auto-detects hardware (CUDA float16/int8 vs CPU int8)[cite: 19]
2. Reads manuscripts/<stop>.txt & lexicon/lexicon.json[cite: 19]
3. Parses inline [IMG: src | caption] tags, recording anchor word indices and stripping tags before TTS/display
4. Synthesizes 96k mono MP3 via Kokoro-82M & FFmpeg[cite: 19]
5. Runs WhisperX forced alignment + difflib global opcode mapping[cite: 19]
6. Resolves [IMG: ...] tags to exact spoken start/end timestamps
7. Generates full-width mobile test_player_v3.html with:
   - Timed Visual Cueing Stage + < / > artifact chapter skipping
   - Lower Thumb Zone: [-10s] [Play/Pause] [+10s] + Audio Timeline Scrubber
   - 50% Collapsible Transcript Drawer ("REVEAL TEXT" / "HIDE TEXT") with dynamic hero compression & tap-to-seek
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

# Dynamic hardware selection[cite: 19]
if torch.cuda.is_available():
    DEVICE = "cuda"
    major_cc = torch.cuda.get_device_capability()[0]
    COMPUTE_TYPE = "float16" if major_cc >= 7 else "int8"
    WHISPER_BATCH_SIZE = 16
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
    WHISPER_BATCH_SIZE = 4

DEFAULT_TEST_MANUSCRIPT = """[IMG: images/vim_knife-x3.webp | Roman Military Knives]
Legionaries stationed along the Danube frontier carried sturdy iron knives as essential multipurpose tools for both camp life and combat. These blades were used for preparing rations, repairing leather gear, and close-quarters defense. The preserved handles and sheaths from Viminacium reflect the standardized craftsmanship of Roman military workshops.

[IMG: images/vim_medicine-x3.webp | Roman Surgical Instruments]
Roman military doctors, known as medici-chirurgi, treated battlefield injuries using specialized bronze and iron instruments. Excavations at Viminacium have uncovered scalpels, tweezers, double-ended probes, and delicate needles designed for intricate procedures. These finds demonstrate an advanced understanding of anatomy and field surgery within the imperial army.

[IMG: images/vim-coins-x2.webp | Roman Imperial Coinage]
Coinage served as the lifeblood of the Roman frontier economy, funding legionary pay and daily trade across Moesia. Silver denarii and bronze sestertii minted under various emperors circulated widely through the military camp and civilian settlement. Beyond their monetary value, the portraits and inscriptions on these coins broadcast imperial authority to every corner of the empire.
"""

# =====================================================================
# 1. LEXICON & TEXT NORMALIZATION[cite: 19]
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
    lex = {"serbian_words": set(), "diacritic_restore": {}, "ipa": {}, "roman_exclude": set()}
    if not os.path.exists(path):
        print(f"[LEXICON] Not found at {path}; using built-in phonetic rules.")
        return lex
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lex["serbian_words"] = {w.lower() for w in data.get("serbian_words", [])}
    lex["diacritic_restore"] = {k.lower(): v for k, v in data.get("diacritic_restore", {}).items()}
    lex["ipa"] = {k.lower(): v for k, v in data.get("ipa", {}).items()}
    lex["roman_exclude"] = {w.lower() for w in data.get("roman_exclude", [])}
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
# 2. INLINE [IMG: ...] PARSER & WORD NORMALIZATION
# =====================================================================
def normalize_word(w: str) -> str:
    t = w.lower()
    replacements = {'ž': 'z', 'č': 'c', 'ć': 'c', 'š': 's', 'đ': 'dj', 'oe': 'e', 'ae': 'e'}
    for src, dst in replacements.items():
        t = t.replace(src, dst)
    return re.sub(r'[^\w]', '', t)

def parse_manuscript_with_images(raw_text: str):
    """
    Extracts [IMG: src | caption] tags, tracks the exact normalized word index
    where each image starts, and returns clean prose paragraphs for TTS & UI.
    """
    img_pattern = re.compile(r'\[IMG:\s*([^|\]]+?)(?:\|\s*([^\]]+?))?\s*\]', re.IGNORECASE)
    
    clean_lines = []
    image_markers = []
    current_word_count = 0

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            clean_lines.append("")
            continue

        pos = 0
        line_clean_parts = []
        for match in img_pattern.finditer(stripped):
            before = stripped[pos:match.start()].strip()
            if before:
                line_clean_parts.append(before)
                current_word_count += sum(1 for w in before.split() if normalize_word(w))
            
            src = match.group(1).strip()
            caption = (match.group(2) or "").strip()
            image_markers.append({
                "word_index": current_word_count,
                "src": src,
                "caption": caption
            })
            pos = match.end()

        after = stripped[pos:].strip()
        if after:
            line_clean_parts.append(after)
            current_word_count += sum(1 for w in after.split() if normalize_word(w))

        joined_line = " ".join(line_clean_parts).strip()
        if joined_line:
            clean_lines.append(joined_line)

    clean_text = "\n".join(clean_lines)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
    return clean_text, image_markers

# =====================================================================
# 3. AUDIO ENCODING & SENTENCE SPLITTING[cite: 19]
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

ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "st", "no", "nos", "vs", "c", "ca", "cf", "fig", "figs",
    "vol", "approx", "ed", "eds", "pp", "p", "jr", "sr", "e.g", "i.e", "viz"
}

def split_sentences(para: str) -> list:
    sentences = []
    start = 0
    for m in re.finditer(r'[.!?]+["”’\)\]]*\s+', para):
        nxt = para[m.end():m.end() + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in '"“‘\(\['):
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
# 4. HTML TEST HARNESS GENERATOR (v3 FULL-WIDTH MOBILE UI)
# =====================================================================
def generate_test_player_html(stop_data, html_path):
    default_img = stop_data['images'][0]['src'] if stop_data['images'] else ''
    default_cap = stop_data['images'][0].get('caption', '') if stop_data['images'] else ''

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>{stop_data.get('title', 'Audio Guide v3 Test')}</title>
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
      background-color: var(--bg-cream);
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100dvh;
      overflow: hidden;
    }}
    .phone-simulator {{
      width: 100%;
      height: 100dvh;
      background-color: var(--bg-cream);
      display: flex;
      flex-direction: column;
      position: relative;
      overflow: hidden;
    }}
    .screen-content {{
      padding: 20px 16px 16px;
      display: flex;
      flex-direction: column;
      width: 100%;
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
      width: 34px; height: 34px;
      background: rgba(0,0,0,0.45);
      border: 1px solid rgba(255,255,255,0.3);
      color: #fff;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.25rem;
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
    .transcript-drawer p {{ margin-bottom: 0.85rem; }}
    .transcript-drawer p:last-child {{ margin-bottom: 0; }}
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
      <img class="stage-img" id="stage-img" src="{default_img}" alt="Artifact">
      <button class="stage-nav next" id="next-artifact" aria-label="Next artifact">&rsaquo;</button>
      <span class="stage-caption" id="stage-caption">{default_cap}</span>
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

  const paraMap = new Map();
  STOP_DATA.cues.forEach((cue) => {{
    const pIdx = cue.p ?? 0;
    if (!paraMap.has(pIdx)) {{
      const pEl = document.createElement('p');
      paraMap.set(pIdx, pEl);
      transcriptBody.appendChild(pEl);
    }}
    const span = document.createElement('span');
    span.className = 'cue';
    span.dataset.start = cue.start;
    span.dataset.end = cue.end;
    span.textContent = cue.text + ' ';
    paraMap.get(pIdx).appendChild(span);
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
# 5. MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Build & Test v3 Stop with full-width mobile UI")
    parser.add_argument("--stop", type=str, default="test_medicine", help="Manuscript filename (without .txt)")
    parser.add_argument("--voice", type=str, default="af_bella", help="Kokoro voice preset")
    args = parser.parse_args()

    stop_id = args.stop
    manuscript_file = os.path.join(MANUSCRIPTS_DIR, f"{stop_id}.txt")
    mp3_filename = f"{stop_id}.mp3"
    mp3_path = os.path.join(OUTPUT_DIR, mp3_filename)
    output_json = os.path.join(OUTPUT_DIR, f"{stop_id}_stop_v3.json")
    output_html = os.path.join(BASE_DIR, "test_player_v3.html")

    if not os.path.exists(manuscript_file):
        with open(manuscript_file, "w", encoding="utf-8") as f:
            f.write(DEFAULT_TEST_MANUSCRIPT)
        print(f"[MANUSCRIPT] Created sample manuscript with [IMG:] tags at: {manuscript_file}")

    with open(manuscript_file, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    # 1. Extract [IMG: ...] markers & strip them from display/TTS text
    clean_text, image_markers = parse_manuscript_with_images(raw_text)
    print(f"[{stop_id.upper()}] Parsed {len(image_markers)} inline visual cue tags.")

    lexicon = load_lexicon()
    normalized_text = process_text(clean_text, lexicon)

    # 2. Kokoro TTS Synthesis[cite: 19]
    print(f"[{stop_id.upper()}] Synthesizing narration via Kokoro-82M ({args.voice}) on {DEVICE}...")
    tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device=DEVICE)
    chunks = [chunk for _, _, chunk in tts_pipeline(normalized_text, voice=args.voice, speed=1.0)]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio.")

    full_audio = np.concatenate(chunks)
    save_mp3(full_audio, mp3_path, sample_rate=SAMPLE_RATE, bitrate=BITRATE)
    print(f"[{stop_id.upper()}] Saved MP3: {mp3_path}")

    # 3. WhisperX Forced Alignment + Global Opcode Matching[cite: 19]
    print(f"[{stop_id.upper()}] Running WhisperX alignment on {DEVICE} ({COMPUTE_TYPE})...")
    paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
    ms_words_norm = []
    sentence_word_ranges = []

    for p_idx, para in enumerate(paragraphs):
        sentences = split_sentences(para)
        for sent in sentences:
            raw_tokens = [w for w in sent.split() if normalize_word(w)]
            start_i = len(ms_words_norm)
            for w in raw_tokens:
                ms_words_norm.append(normalize_word(w))
            end_i = len(ms_words_norm) - 1
            sentence_word_ranges.append((p_idx, sent, start_i, end_i))

    audio = whisperx.load_audio(mp3_path)
    audio_duration = round(len(audio) / 16000.0, 2)

    whisper_model = whisperx.load_model("small.en", DEVICE, compute_type=COMPUTE_TYPE)
    result = whisper_model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE)

    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
    aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, DEVICE, return_char_alignments=False)

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
    for p_idx, sent_text, start_i, end_i in sentence_word_ranges:
        cues.append({
            "p": p_idx,
            "start": round(ms_word_start[start_i], 2),
            "end": round(ms_word_end[end_i], 2),
            "text": sent_text
        })
    if cues and cues[-1]["end"] < cues[-1]["start"]:
        cues[-1]["end"] = audio_duration

    # 4. Resolve [IMG: ...] word indices to exact timestamps
    images = []
    for idx, marker in enumerate(image_markers):
        w_idx = min(marker["word_index"], len(ms_word_start) - 1)
        start_t = 0.0 if idx == 0 and w_idx == 0 else round(ms_word_start[w_idx], 2)
        images.append({
            "start": start_t,
            "end": audio_duration,
            "src": marker["src"],
            "caption": marker["caption"]
        })

    for idx in range(len(images) - 1):
        images[idx]["end"] = images[idx + 1]["start"]

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
    print(f"\n[DONE] Built full-width mobile player ({audio_duration}s) with {len(images)} inline image triggers.")
    print("Open: http://localhost:8000/test_player_v3.html")

if __name__ == "__main__":
    main()