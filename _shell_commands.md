Activate local venv

.\venv\Scripts\activate

    ACTIVATE KOKORO TTS:
gui_tts.py

    local python hosting:
python -m http.server 8000

    PYTHON LOCAL HOST ADDRESS
http://localhost:8000/test_player.html

    FORCED ALIGNMENT DEMO
python align_demo.py