# Workstation Environment Setup Guide

This guide covers setting up the development and offline audio production environment for the **Audio Guide SPA** on a new Windows computer.

---

## 1. System Prerequisites

Open **PowerShell as Administrator** and install Python 3.12, FFmpeg, and Git via Windows Package Manager (`winget`):

```powershell
# Install Python 3.12
winget install Python.Python.3.12

# Install FFmpeg (required for audio conversion and WhisperX)
winget install Gyan.FFmpeg

# Install Git (if not already installed)
winget install Git.Git
```

> **Note:** Close and reopen your PowerShell terminal after installation so your system environment variable `$env:PATH` refreshes.

---

## 2. Configure PowerShell Script Execution Policy

Windows blocks script execution by default. Configure your user profile to allow local virtual environment scripts to run permanently without requiring manual bypass commands every session:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
*When prompted, press `Y` (or `A` for "Yes to All").*

---

## 3. (Recommended) Add Windows Security Folder Exclusion

To prevent antivirus latency, file locks on virtual environment binaries, or interference with AI model weights, add the project folder to Windows Defender exclusions:

```powershell
Add-MpPreference -ExclusionPath "C:\audio_guide_SPA"
```

---

## 4. Clone the Repository

Clone the project to a local directory directly on your drive (avoid OneDrive or cloud sync paths to prevent file locking and path conflicts):

```powershell
cd C:\
git clone https://github.com/<your-username>/<your-repo-name>.git audio_guide_SPA
cd audio_guide_SPA
```

---

## 5. Initialize the Virtual Environment

Create an isolated virtual environment locked to Python 3.12 in the root of the project:

```powershell
# Create venv
py -3.12 -m venv venv

# Activate venv
.\venv\Scripts\activate
```

When active, your terminal prompt will be prefixed with `(venv)`:
```text
(venv) PS C:\audio_guide_SPA>
```

*(To exit the virtual environment at any point, simply run `deactivate`).*

---

## 6. Install Project Dependencies

With `(venv)` active, restore all required libraries (Kokoro TTS, WhisperX, PyTorch, soundfile, etc.) from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

---

## 7. Verify Module Installations

Confirm that both speech synthesis and forced alignment models import cleanly:

```powershell
# Verify Kokoro TTS and audio drivers
python -c "import kokoro, soundfile; print('Kokoro ready!')"

# Verify WhisperX forced alignment
python -c "import whisperx; print('WhisperX ready!')"
```

---

## 8. Common Troubleshooting

### Script Execution Disabled Error
If running `.\venv\Scripts\activate` throws an `UnauthorizedAccess` error:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate
```

### Windows 11 Smart App Control
If Python throws `Application Control policy has blocked this file` when importing C-extensions (`srsly`, `ctranslate2`):
* Navigate to **Windows Settings > Privacy & security > Windows Security > App & browser control**.
* Turn off or evaluate **Smart App Control**.

### First Model Run Initialization
The first time a synthesis or alignment script executes, weights are downloaded to your local cache (`~/.cache/huggingface/`):
* Kokoro model weights: `kokoro-v1_0.pth` (~327 MB)
* WhisperX alignment model: `wav2vec 2.0`
* Future pipeline runs will operate 100% locally and offline.