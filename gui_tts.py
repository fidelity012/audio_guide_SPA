import os
import re
import shutil
import subprocess
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import soundfile as sf
from kokoro import KPipeline

# Silence internal PyTorch / HF Hub warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# =====================================================================
# 0. DIRECTORY & ENCODING CONFIGURATION
# =====================================================================
OUTPUT_DIR = r"C:\Users\marko\OneDrive\_DEV\audio_vodic\audio_guide_SPA\audio_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEFAULT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.mp3")

BITRATE = "96k"
SAMPLE_RATE = 24000

# =====================================================================
# 1. COMPREHENSIVE VOCABULARY & PHONETIC RULES
# =====================================================================

ROMAN_ORDINALS = {
    'I': 'First', 'II': 'Second', 'III': 'Third', 'IV': 'Fourth',
    'V': 'Fifth', 'VI': 'Sixth', 'VII': 'Seventh', 'VIII': 'Eighth',
    'IX': 'Ninth', 'X': 'Tenth', 'XI': 'Eleventh', 'XII': 'Twelfth',
    'XIII': 'Thirteenth', 'XIV': 'Fourteenth', 'XV': 'Fifteenth',
    'XVI': 'Sixteenth', 'XVII': 'Seventeenth', 'XVIII': 'Eighteenth',
    'XIX': 'Nineteenth', 'XX': 'Twentieth'
}

# Serbian proper nouns without diacritics
CUSTOM_SERBIAN_WORDS = {
    "belovode", "kostolac", "vlasac", "padina", "dubravica", "drmno",
    "golubac", "dubovac", "klenovnik", "usije", "salakovac", "trnjane",
    "kasidol", "klenje", "leskovac", "tabanovac", "krvije", "stubline",
    "drenovac", "vatin", "gava", "brdo",
    "morava", "mlava", "busur", "beljanica",
    "krstic", "jacanovic", "sljivar"
}

# Latin archaeological & ancient terms (clean phonemes without stress markers)
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
    """Expands Roman numerals for military legions and emperors."""
    def replace_legion_pair(m):
        prefix = m.group(1)
        n1 = m.group(2).upper()
        conn = m.group(3)
        n2 = m.group(4).upper()
        return f"{prefix} {ROMAN_ORDINALS.get(n1, n1)} {conn} {ROMAN_ORDINALS.get(n2, n2)}"

    text = re.sub(r'\b(legions?)\s+([IVXLCDM]+)\s+(and|&)\s+([IVXLCDM]+)\b', replace_legion_pair, text, flags=re.IGNORECASE)

    def replace_legion(m):
        prefix = m.group(1)
        num = m.group(2).upper()
        return f"{prefix} {ROMAN_ORDINALS.get(num, num)}"

    text = re.sub(r'\b(legions?)\s+([IVXLCDM]+)\b', replace_legion, text, flags=re.IGNORECASE)

    rulers = (
        r'(?:Gordian|Antoninus|Hadrian|Justinian|Trajan|Severus|'
        r'Merneptah|Ramesses|Constantine|Diocletian|Claudius|Marcus\s+Aurelius)'
    )
    def replace_ruler(m):
        ruler = m.group(1)
        num = m.group(2).upper()
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
    """
    1. Expands Roman numerals (Legion VII -> Legion Seventh).
    2. Injects Latin / Ancient terms with smooth phonemes.
    3. Converts Serbian proper nouns to inline Kokoro IPA tags.
    """
    text = expand_roman_numerals(text)

    for term, ipa_override in SPECIAL_TERMS.items():
        pattern = rf'\b{term}\b'
        text = re.sub(pattern, ipa_override, text, flags=re.IGNORECASE)

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
# 2. AUDIO EXPORT (MP3: 96k, Mono, 24kHz)
# =====================================================================

def save_mp3(audio_data: np.ndarray, output_path: str, sample_rate: int = SAMPLE_RATE, bitrate: str = BITRATE):
    """Encodes raw NumPy audio to 96k Mono MP3 at 24 kHz."""
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
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _, stderr = proc.communicate(input=pcm_bytes)
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"FFmpeg error: {err}")
    else:
        try:
            sf.write(output_path, audio_data, sample_rate, format="MP3")
        except Exception:
            fallback_wav = os.path.splitext(output_path)[0] + ".wav"
            sf.write(fallback_wav, audio_data, sample_rate)
            raise RuntimeError(
                f"FFmpeg was not detected in PATH. Audio was saved as {fallback_wav}.\n"
                "To enable 96k MP3 encoding, run: winget install Gyan.FFmpeg"
            )

# =====================================================================
# 3. SYNTHESIS ENGINE
# =====================================================================

print("Initializing Kokoro TTS...")
pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

def synthesize(text: str, output_path: str = DEFAULT_OUTPUT_PATH, voice: str = "af_bella"):
    processed = process_text(text)
    chunks = []
    for _, _, audio in pipeline(processed, voice=voice, speed=1.0):
        chunks.append(audio)

    if chunks:
        full_audio = np.concatenate(chunks)
        save_mp3(full_audio, output_path, sample_rate=SAMPLE_RATE, bitrate=BITRATE)
    return processed

# =====================================================================
# 4. GUI INTERFACE
# =====================================================================

def on_generate():
    raw_text = input_box.get("1.0", tk.END).strip()
    if not raw_text:
        messagebox.showwarning("Warning", "Please enter some text first.")
        return

    btn_generate.config(state=tk.DISABLED, text="Synthesizing & Encoding MP3...")
    root.update()

    try:
        selected_voice = voice_var.get()
        processed = synthesize(raw_text, DEFAULT_OUTPUT_PATH, selected_voice)

        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, processed)

        status_lbl.config(text=f"Saved MP3 (96k mono): {DEFAULT_OUTPUT_PATH}", foreground="#1b7a2b")
        btn_play.config(state=tk.NORMAL)
    except Exception as e:
        status_lbl.config(text=f"Error: {str(e)}", foreground="#b00020")
    finally:
        btn_generate.config(state=tk.NORMAL, text="Convert to IPA & Generate Audio")

def on_play():
    if os.path.exists(DEFAULT_OUTPUT_PATH):
        os.startfile(DEFAULT_OUTPUT_PATH)

root = tk.Tk()
root.title("Kokoro Museum Audio Guide Studio (MP3 Output)")
root.geometry("740x680")

ttk.Label(root, text="1. Exhibition Script (English with Serbian proper nouns & Roman Legions):", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
input_box = tk.Text(root, height=10, wrap="word", font=("Segoe UI", 10))
input_box.pack(fill="x", padx=14)

ctrl_frame = ttk.Frame(root)
ctrl_frame.pack(fill="x", padx=14, pady=8)

ttk.Label(ctrl_frame, text="Voice:").pack(side="left", padx=(0, 5))
voice_var = tk.StringVar(value="af_bella")
voice_selector = ttk.Combobox(ctrl_frame, textvariable=voice_var, values=["af_bella", "af_sarah", "bf_emma", "bm_george"], width=12, state="readonly")
voice_selector.pack(side="left", padx=(0, 15))

btn_generate = ttk.Button(ctrl_frame, text="Convert to IPA & Generate Audio", command=on_generate)
btn_generate.pack(side="left", fill="x", expand=True, padx=(0, 10))

btn_play = ttk.Button(ctrl_frame, text="Play MP3", command=on_play, state=tk.DISABLED)
btn_play.pack(side="right")

ttk.Label(root, text="2. Processed Script Sent to Kokoro (Inspect Roman numerals & IPA tags):", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
output_box = tk.Text(root, height=8, wrap="word", font=("Segoe UI", 9), background="#f8f9fa")
output_box.pack(fill="x", padx=14)

status_lbl = ttk.Label(root, text=f"Output File: {DEFAULT_OUTPUT_PATH} (96k mono, 24kHz)", font=("Segoe UI", 8))
status_lbl.pack(pady=10)

root.mainloop()