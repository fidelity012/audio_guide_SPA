# SPEC.md: Reusable Museum Audio Guide Engine

## 1. System Overview & Core Principles

* **Decoupled Architecture:** The system is split strictly into two distinct domains: an offline Python batch content pipeline and a client-side static web presentation engine.
* **Serverless & Zero Backend:** The production audio guide operates entirely on static edge hosting (GitHub Pages or Cloudflare Pages). No application server, database, user authentication, or runtime speech APIs exist in production.
* **White-Label Reusability:** Core markup, styles, and JavaScript logic are institution-agnostic. Deploying a new museum requires altering only the `/content/` directory, never the core application engine.
* **Zero-Install Physical Onboarding:** Visitors access exhibit content by scanning physical QR stands placed at gallery entrances. Navigation between exhibition rooms occurs via URL hashes (`#prehistory`, `#viminacium`, `#religion`) without installing native apps.
* **Heads-Up Visitor UX & Visual Cueing:** Physical exhibits remain the primary focus over smartphone screens. Instead of continuous text scrolling, the screen acts as a synchronized visual viewfinder that swaps between showcase overviews and macro object photos as the narrator speaks. Transcripts remain tucked away by default inside an on-demand, half-screen collapsible drawer with active sentence bookmarking and tap-to-seek scrubbing.

---

## 2. Standardized Directory Layout

```text
audio_guide_SPA/
├── SPEC.md                      # Authoritative system specification & contracts
├── README.md                    # Quick-start and workstation setup instructions
├── .gitignore                   # Excludes virtual environments, caches, and raw audio
├── requirements.txt             # Frozen Python dependencies for reproduction
│
├── index.html                   # Generic SPA entry shell (root-level for edge hosting)
├── guide.js                     # Router, Swiper carousel manager, audio sync engine
├── style.css                    # Layout rules, mobile viewport gate, and CSS variables
├── test_player.html             # Single-stop local verification harness
├── vendor/                      # Optional offline vendor assets (Swiper JS/CSS)
│
├── content/                     # MUSEUM PAYLOAD (Swappable per institution)
│   ├── manifest.json            # Gallery metadata, stop definitions, images, and cues
│   ├── theme.json               # Visual identity tokens (palette, fonts, branding)
│   ├── lexicon.json             # Museum-specific Serbian words, diacritics, and IPA
│   ├── audio/                   # Production audio (96 kbps mono MP3s, 24 kHz)
│   └── img/                     # Production artifact imagery (WebP, max-width 350px)
│
└── pipeline/                    # OFFLINE CONTENT FACTORY (Python batch tooling)
    ├── manuscripts/             # Raw curator input manuscripts (.txt)
    ├── build_all_stops.py       # Master batch pipeline (Singleton Kokoro + WhisperX)
    ├── build_stop.py            # Single-stop headless build & verification runner
    ├── serbian_to_ipa.py        # Deterministic Latin-to-IPA phonetic rules
    └── audio_output/            # Intermediate build outputs and debug *_cues.json files
```

---

## 3. Data Contracts & Schemas

### 3.1 Exhibition Manifest (`content/manifest.json`)
The manifest represents the single source of truth for exhibition hierarchy, media asset locations, timed visual cue stages, and synchronized sentence cues. Each stop defines an `images` array with time boundaries (`start`, `end`) to drive visual cueing, alongside sentence `cues` with paragraph indices (`p`).

```json
{
  "museum": {
    "id": "nm-pozarevac",
    "name": "National Museum of Požarevac – Audio Guide",
    "defaultRoom": "prehistory",
    "roomAliases": {
      "late-antiquity": "religion"
    }
  },
  "rooms": [
    {
      "slug": "viminacium",
      "title": "Roman Antiquities & Viminacium",
      "endNotice": "End of Viminacium section. Please scan the QR code in the next gallery to continue.",
      "stops": [
        {
          "id": "2_4_roman_medicine",
          "room": "viminacium",
          "title": "Roman Medicine in Viminacium",
          "subtitle": "Showcase 4: Surgical Practice",
          "manuscript": "2_4_roman_medicine.txt",
          "audio": "content/audio/2_4_roman_medicine.mp3",
          "images": [
            {
              "start": 0.0,
              "end": 22.0,
              "src": "content/img/med_vitrine_overview.webp",
              "caption": "Showcase 4: Overview of Surgical Instruments"
            },
            {
              "start": 22.0,
              "end": 51.5,
              "src": "content/img/bronze_box_detail.webp",
              "caption": "Bronze Instrumentarium & Probes"
            },
            {
              "start": 51.5,
              "end": 94.0,
              "src": "content/img/cataract_needle.webp",
              "caption": "Specialized Cataract Needle for Ocular Surgery"
            }
          ],
          "cues": [
            {
              "p": 0,
              "start": 0.0,
              "end": 6.8,
              "text": "Viminacium, a major Roman city and military camp on the Danube, had a sophisticated medical community."
            },
            {
              "p": 0,
              "start": 6.85,
              "end": 14.2,
              "text": "The medicus-chirurgus was buried with a rich set of surgical tools, including tweezers, probes, and a bronze instrumentarium box."
            },
            {
              "p": 1,
              "start": 14.5,
              "end": 21.8,
              "text": "Excavated graves in the southern necropolis have yielded various specialised instruments."
            }
          ]
        }
      ]
    }
  ]
}
```

### 3.2 Visual Theme Tokens (`content/theme.json`)
Theme values are injected directly into `:root` CSS custom properties upon initialization.

```json
{
  "primaryAccent": "#C65D3B",
  "backgroundColor": "#F5F2ED",
  "textColor": "#2E2E2E",
  "cardBackground": "#FFFFFF",
  "fontHeading": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  "fontBody": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
}
```

### 3.3 Pronunciation Lexicon (`content/lexicon.json`)
Isolates regional toponyms, surnames, and classical loanwords from the Python codebase. All keys are lowercase; visitor display text is never altered.

```json
{
  "_readme": "Per-museum pronunciation lexicon. Keys are lowercase. Display text is never changed.",
  "language": "sr",
  "serbian_words": [
    "belovode",
    "kostolac",
    "vlasac",
    "padina",
    "dubravica",
    "drmno",
    "golubac",
    "dubovac",
    "morava",
    "mlava",
    "busur",
    "beljanica",
    "ram"
  ],
  "diacritic_restore": {
    "jacanovic": "jacanović",
    "krstic": "krstić",
    "sljivar": "šljivar"
  },
  "ipa": {
    "viminacium": "viːmiːnɑːtsiːuːm",
    "moesia": "miːʃə",
    "margum": "mɑːrɡʊm",
    "lederata": "lɛdɛrɑːtɑː",
    "pincum": "pɪŋkʊm",
    "cuppae": "kʊpaɪ",
    "novae": "noʊvaɪ",
    "aesculapius": "iːskjuːlɑːpiːʊs",
    "chirurgus": "kiːrʊrɡʊs",
    "ocularius": "oʊkjuːlɑːriːʊs",
    "steatopygous": "stiːətəʊpaɪɡəs"
  }
}
```

---

## 4. Media & Asset Standards

* **Audio Encoding:** Constant Bitrate (CBR) MP3, 96 kbps, mono channel (`-ac 1`), 24,000 Hz sample rate (matching Kokoro-82M native output), piped directly via FFmpeg (`f32le` PCM input to `libmp3lame`).
* **Image Format:** WebP format, max file size 150 KB. Images are presented inside a responsive container (`.slide-media-stage`) with default height `12rem` and dynamic compression down to `6.5rem` when the transcript drawer is revealed.
* **Version Control Boundaries:** Production MP3s (`content/audio/*.mp3`) and WebP images (`content/img/*.webp`) are tracked in Git; raw `.wav` recordings, transient test audio (`test_*.mp3`, `output.wav`), and neural model weights remain strictly `.gitignore`d.

---

## 5. Offline Content Pipeline Specification

```text
[Manuscript .txt] ──► [Normalizer & Lexicon] ──► [Kokoro-82M TTS] ──► [FFmpeg 96k MP3]
       │                                                                       │
       └─────────────────────────► [WhisperX wav2vec 2.0] ◄────────────────────┘
                                           │
                                           ▼
                             [difflib Global Opcode Map]
                                           │
                                           ▼
                      [audio_output/*_cues.json + manifest.json]
```

### 5.1 Hardware Auto-Detection
To support multi-PC development without manual code edits, the pipeline dynamically selects hardware at startup:
* **NVIDIA CUDA GPU Present (e.g., RTX 3060 Ti):** `DEVICE = "cuda"`, `COMPUTE_TYPE = "float16"`.
* **CPU Fallback (e.g., AMD Ryzen 7 260):** `DEVICE = "cpu"`, `COMPUTE_TYPE = "int8"`.

### 5.2 Pipeline Execution Stages
1. **Manuscript Ingestion:** Reads UTF-8 manuscripts from `pipeline/manuscripts/`. Tokenizes sentences via lookbehind regex (`(?<=[.!?])\s+`) to preserve abbreviations and paragraph index (`p`).
2. **Three-Tier Phonetic & Text Normalization:**
   * **Roman Numerals:** Expands military legions (*legion VII* $\rightarrow$ *legion Seventh*) and rulers (*Gordian III* $\rightarrow$ *Gordian the Third*).
   * **Explicit IPA Overrides:** Matches words in `lexicon.json["ipa"]` and wraps them in Kokoro inline syntax `[Word](/ipa/)`.
   * **Serbian Orthography & Diacritics:** Applies `diacritic_restore`, then converts words with Serbian diacritics (`[čćžšđČĆŽŠĐ]`) or terms in `serbian_words` using `serbian_to_kokoro_ipa()`.
3. **Speech Synthesis:** Initializes `KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')` as a singleton and synthesizes speech chunks at `speed=1.0`.
4. **MP3 Encoding:** Concatenates NumPy float32 audio chunks and pipes them through FFmpeg into a 96 kbps mono MP3 at 24 kHz.
5. **Acoustic Transcription & Forced Alignment:** WhisperX (`small.en`) transcribes audio and executes `wav2vec 2.0` phoneme alignment to extract word-level timestamps.
6. **Global Opcode Sequence Matching:** Normalizes tokens (stripping punctuation and mapping Slavic diacritics) and aligns manuscript words against acoustic words using `difflib.SequenceMatcher.get_opcodes()`:
   * `equal`: Direct 1:1 timestamp assignment.
   * `replace`: Proportional linear interpolation across the replaced span.
   * `delete`: Clamps to the adjacent acoustic word boundary.
7. **Sentence Boundary Clamping:** Derives sentence start and end times (rounded to 2 decimal places). Clamps final end time to total audio duration.
8. **Dual Output Delivery:** Generates `pipeline/audio_output/<stop>_cues.json` for test harness verification and updates `content/manifest.json` for production.

---

## 6. Frontend SPA Engine Specification

### 6.1 Viewport & Layout Architecture
* **Mobile-First Viewport Gate:** `style.css` restricts display to mobile screens (`@media (min-width: 601px) { body { display: none !important; } }`).
* **Ergonomic "Thumb Zone" Organization:** Primary interactive controls are clustered in the lower half of the screen for one-handed operation while standing or walking through galleries.

```text
┌──────────────────────────────────────────────┐
│             ROMAN MEDICINE                   │  <── Stop Title
│                                              │
│   ┌──────────────────────────────────────┐   │
│   │                                      │   │
│   │  <        [ Artifact Photo ]       > │   │  <── Timed Visual Cue Stage
│   │                                      │   │      (< / > skips artifacts)
│   └──────────────────────────────────────┘   │
│                                              │
│          (↺ 10)     (||)     (10 ↻)          │  <── Playback & Micro-Skip Cluster
│         0:34 ───────●───────── 1:30          │  <── Audio Scrubber Bar
│                                              │
│             [ REVEAL TEXT  ▼ ]               │  <── Collapsible Transcript Toggle
├──────────────────────────────────────────────┤  ▲
│ The medicus-chirurgus was buried with a rich │  │
│ set of surgical tools, including tweezers,   │  │ 45%-50% Viewport Drawer
│ ◀ probes, and a bronze instrumentarium box.  │  │ (Active sentence highlighted)
│ Excavated graves in the southern necropolis  │  │ (Internal vertical scroll)
│ have yielded various specialised instruments.│  ▼
└──────────────────────────────────────────────┘
```

### 6.2 Visual Cueing & Artifact Navigation
1. **Timed Image Swapping:** The hero image stage switches automatically during audio playback based on timestamps defined in `stop.images`. Transitions use an opacity crossfade (250ms) to prevent abrupt visual jumps.
2. **Artifact Sub-Topic Skipping (`<` and `>` chevrons):**
   * The chevrons on the image stage act as macro chapter jumps between artifacts within the vitrine:
     * **`>` (Next Artifact):** Advances `audio.currentTime` to the next image's `start` timestamp and calls `audio.play()`.
     * **`<` (Previous Artifact):** If playback is more than 3 seconds into the current artifact, restarts that artifact; otherwise jumps back to the preceding artifact.
3. **Gallery Stop Navigation vs. Audio Controls:** Moving between physical exhibition stops is handled strictly via **horizontal Swiper gestures** (or margin arrows). "Next Stop" buttons must never be mixed into the audio playback cluster to avoid accidental skips.

### 6.3 Collapsible Transcript Drawer & Background Sync
1. **Default State (Collapsed):** The hero image stage displays at full height (`12rem`). The transcript drawer remains hidden. The button displays `"REVEAL TEXT ▼"`.
2. **Background Sync:** The audio `timeupdate` event updates `.cue.active` classes in memory at all times, even while the drawer is closed. There is zero latency upon opening.
3. **Open State (Revealed):**
   * The button updates to `"HIDE TEXT ▲"`.
   * The transcript drawer opens to occupy **45% to 50% of viewport height (`max-height: 48vh`)** with internal scrolling (`overflow-y: auto`).
   * **Dynamic Height Compression:** The hero image container compresses down to `6.5rem` via CSS transition, ensuring stop title, photo, audio scrubber, and transcript are all visible in a single viewport without page scrolling.
   * **Auto-Scroll to Active Bookmark:** When toggled open, the drawer automatically executes `activeCue.scrollIntoView({ behavior: 'smooth', block: 'center' })` to center the current spoken sentence.
4. **Lifecycle & Reset Rules:**
   * The drawer does **not** auto-close when audio playback ends, allowing visitors to finish reading or rewind comfortably.
   * The drawer resets to collapsed (`text-expanded` removed) whenever the visitor swipes to another slide or switches rooms.

---

## 7. Inviolable Architectural Rules & Anti-Patterns

1. **Ground-Truth Display Invariance:** Whisper transcription text is strictly transient and must **never** be rendered on screen. The visitor screen must display only clean curator manuscripts with full diacritics.
2. **Deterministic Sequence Alignment:** Always map audio timestamps back onto authentic manuscript words using `difflib.get_opcodes`. Never use naive word-count division.
3. **Stress Ticks in Kokoro IPA:** Never pass Unicode primary stress ticks (`ˈ`) into Kokoro inline `[Word](/ipa/)` tags.
4. **Single-Track Enforcement:** Never allow two audio tracks to play simultaneously. Swiping slides or changing rooms must immediately pause active audio.
5. **No Mixed Navigation Levels:** Do not place "Next Stop" buttons inside the audio playback controls. Micro-controls ($\pm$10s) belong to audio; carousel arrows belong to sub-topic artifacts; Swiper swipes belong to gallery stops.

---

## 8. New Museum Onboarding Runbook

1. **Template Clone:** Create a new repository from template and configure `content/theme.json` with the institution's colors and fonts.
2. **Asset & Manuscript Ingestion:** Place UTF-8 curator manuscripts in `pipeline/manuscripts/*.txt` and compressed WebP photos (max-width: 350px, <150 KB) in `content/img/`.
3. **Skeleton Manifest Setup:** Define rooms, stop titles, image chapters (`start`, `end`, `src`), and manuscript filenames in `content/manifest.json`.
4. **Lexicon Tuning & Batch Build:** Run `python pipeline/build_stop.py --check` to flag unknown proper nouns, update `content/lexicon.json`, and run `python pipeline/build_all_stops.py` to synthesize MP3s and inject WhisperX cues into `content/manifest.json`.
5. **Deployment & QR Generation:** Deploy static files to GitHub Pages or Cloudflare Pages and print room entry QR stands pointing to `https://<domain>/#<room-slug>`.