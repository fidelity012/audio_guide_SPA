# Shell Commands

### Navigate to working directory:

```powershell
cd C:\audio_guide_SPA
```

### Option A: Run directly using venv Python (Recommended - bypasses script execution policies):

```powershell
        .\venv\Scripts\python.exe gui_tts.py
```

### Option B: Activate virtual environment & run:

If PowerShell blocks `Activate.ps1` (`PSSecurityException` / "running scripts is disabled"):

```powershell
# Allow script execution for your user account (run once):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate and run:
.\venv\Scripts\activate
python gui_tts.py
```
