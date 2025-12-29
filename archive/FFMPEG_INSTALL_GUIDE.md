# FFmpeg Installation Guide for Windows

## Why FFmpeg is Required

OpenAI's Whisper library uses FFmpeg internally to:
- Resample audio files to 16kHz (Whisper's required sample rate)
- Convert various audio formats to a standardized format
- Process audio streams efficiently

Even when saving audio as WAV files, Whisper calls FFmpeg via subprocess for preprocessing.

## Installation Steps

### Method 1: Using Chocolatey (Recommended - Easiest)

If you have Chocolatey package manager installed:

```powershell
choco install ffmpeg
```

### Method 2: Manual Installation (Most Common)

1. **Download FFmpeg**
   - Visit: https://github.com/GyanD/codexffmpeg/releases
   - Download the latest `ffmpeg-git-essentials.zip` or `ffmpeg-git-full.zip`
   - The "essentials" version is sufficient for Whisper

2. **Extract the Archive**
   - Extract the ZIP file to a permanent location
   - Recommended: `C:\ffmpeg`
   - Your folder structure should look like:
     ```
     C:\ffmpeg\
     ├── bin\
     │   ├── ffmpeg.exe
     │   ├── ffplay.exe
     │   └── ffprobe.exe
     ├── doc\
     └── presets\
     ```

3. **Add to System PATH**

   **Option A: Using GUI**
   - Right-click "This PC" → Properties
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find and select "Path"
   - Click "Edit"
   - Click "New"
   - Add: `C:\ffmpeg\bin` (or your actual path)
   - Click "OK" on all dialogs

   **Option B: Using PowerShell (Admin)**
   ```powershell
   # Run PowerShell as Administrator
   setx /m PATH "C:\ffmpeg\bin;%PATH%"
   ```

4. **Verify Installation**
   - **Important**: Close and reopen your terminal/PowerShell
   - Run:
     ```powershell
     ffmpeg -version
     ```
   - You should see version information like:
     ```
     ffmpeg version N-109660-g7e4f32f3f4-20221215 Copyright (c) 2000-2022 the FFmpeg developers
     ...
     ```

### Method 3: Using Scoop Package Manager

If you have Scoop installed:

```powershell
scoop install ffmpeg
```

## Troubleshooting

### "ffmpeg is not recognized as an internal or external command"

**Causes:**
1. FFmpeg not installed
2. Not added to PATH
3. Terminal not restarted after PATH change

**Solutions:**
1. Verify FFmpeg exists at `C:\ffmpeg\bin\ffmpeg.exe`
2. Check PATH contains `C:\ffmpeg\bin`:
   ```powershell
   $env:PATH -split ';' | Select-String ffmpeg
   ```
3. Restart your terminal/IDE completely
4. If in VS Code, reload the window (Ctrl+Shift+P → "Reload Window")

### Virtual Environment Issues

If you're using a virtual environment (like `zuqo`):
- Virtual environments inherit the system PATH
- FFmpeg must be installed globally (not in the venv)
- After installing FFmpeg, deactivate and reactivate your venv:
  ```powershell
  deactivate
  .\zuqo\Scripts\activate
  ```

### Permission Issues

If you get "Access Denied" when modifying PATH:
1. Run PowerShell/Command Prompt as Administrator
2. Or modify User PATH instead of System PATH (works for current user only)

### Still Not Working?

1. **Verify FFmpeg executable:**
   ```powershell
   where.exe ffmpeg
   ```
   Should return: `C:\ffmpeg\bin\ffmpeg.exe`

2. **Test FFmpeg directly:**
   ```powershell
   C:\ffmpeg\bin\ffmpeg.exe -version
   ```

3. **Check Python can find it:**
   ```python
   import shutil
   print(shutil.which("ffmpeg"))
   ```
   Should print the path to ffmpeg.exe

4. **Restart your computer** (last resort, ensures all PATH changes take effect)

## Alternative: Portable FFmpeg

If you can't modify system PATH:

1. Download FFmpeg as above
2. Place `ffmpeg.exe` in your project directory (`d:\python\zuqo\`)
3. Modify the script to use local FFmpeg:
   ```python
   # Add before Whisper initialization
   import os
   os.environ["PATH"] = os.path.dirname(__file__) + os.pathsep + os.environ["PATH"]
   ```

## Verification After Installation

Run this in Python to confirm:

```python
import shutil
import subprocess

# Check if FFmpeg is available
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    print(f"✓ FFmpeg found at: {ffmpeg_path}")
    
    # Get version
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    print(result.stdout.split('\n')[0])
else:
    print("✗ FFmpeg NOT found in PATH")
```

## Next Steps

After installing FFmpeg:
1. Close all terminals and IDEs
2. Reopen your terminal
3. Activate your virtual environment: `.\zuqo\Scripts\activate`
4. Run the test script: `python test.py`

The pipeline should now complete successfully!
