#### 2.11.2026.
#### Audio timeline fix
##### html5 player needs to be refreshed a few times for player bar to work properly.

Cause:  

```python
-m http.server
```
 has no Range support, so the player bar needed 1-2 refreshes.

<p> fix: server with Range requests support: </p>

```js
npx http-server -p 8000 -c-1
```

note: Hosting on GitHub Pages: supports Range requests, so this problem shouldn’t reappear.

#### 4.11.2026.
#### Rendering transition to CUDA      
**PyTorch** exists in all of your environments, `torch.cuda.is_available()` evaluates safely on any machine:
It means if PC doesn't have CUDA, it will use CPU.

Both Work and Home PC's will benefit.

Updates needed:
- **build_stop.py** script
- local python **venv**'s

1 **build script update:**

```py
# Dynamic hardware selection for all 3 PCs
if torch.cuda.is_available():
    DEVICE = "cuda"
    # Ampere+ (RTX 3060 Ti, CC >= 7) 
    # uses native float16; 
    # Pascal (GTX 1080 Ti, CC 6.1) uses int8
    major_cc = torch.cuda.get_device_capability()[0]
    COMPUTE_TYPE = "float16" if major_cc >= 7 else "int8"
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
```

2 **installed py-torch:**

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

#### 10. 11.2026.

While reading the script and question weather to show or hide text, I realized that I should update UX with images as guide topic cues, swipable for progresing trough audio timeline and toggling text visibility.
