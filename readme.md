````md
# faceswap_runpod_setup.md

# GPU Face Swap Setup (RunPod + InsightFace + Web Executor)

## Environment Used

RunPod GPU Pod  
1x RTX A4000 (16GB VRAM)  
Ubuntu 24.04  
Python 3.12  
onnxruntime-gpu 1.24.2  

---

# 1️⃣ Initial Environment Setup

## Update pip

```bash
pip install --upgrade pip
````

## Install Required Libraries

```bash
pip install numpy<2
pip install insightface
pip install onnxruntime-gpu
pip install opencv-python
pip install gradio
pip install tqdm
```

---

# 2️⃣ Verify GPU Is Available

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected output:

```
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

If `CUDAExecutionProvider` is missing → GPU not configured properly.

---

# 3️⃣ Download inswapper_128 Model (Critical)

The auto-download inside subprocess failed.
We manually hosted and downloaded the model.

## Dropbox Model Link Used

```
https://www.dropbox.com/scl/fi/65wx4ew7ehf5fg3cznxgm/inswapper_128.onnx?rlkey=kidd3hb8dkpkcolzcp8tcdk5i&dl=1
```

## Download Command

```bash
curl -L -o ~/.insightface/models/inswapper_128.onnx \
"https://www.dropbox.com/scl/fi/65wx4ew7ehf5fg3cznxgm/inswapper_128.onnx?rlkey=kidd3hb8dkpkcolzcp8tcdk5i&dl=1"
```

## Verify File Size

```bash
ls -lh ~/.insightface/models
```

Expected:

```
inswapper_128.onnx  ~550M
```

If file is small (<1MB), download failed.

---

# 4️⃣ Web Executor App (Upload ZIP + Image + Script + Live Logs)

Create **app.py**

```python
import gradio as gr
import os
import shutil
import subprocess
import tempfile
import time

# ==========================================================
# CONFIG
# ==========================================================

WORK_DIR = "/tmp/faceswap_workspace"

# ==========================================================
# SCRIPT EXECUTION FUNCTION
# ==========================================================

def run_user_script(zip_file, image_file, script_text):

    logs = ""

    # --------------------------------------
    # Prepare workspace
    # --------------------------------------

    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)

    os.makedirs(WORK_DIR)

    logs += "Workspace prepared.\n"
    yield logs, None

    # --------------------------------------
    # Save uploaded ZIP
    # --------------------------------------

    try:
        zip_path = os.path.join(WORK_DIR, "input.zip")
        shutil.copy(
            zip_file.name if hasattr(zip_file, "name") else zip_file,
            zip_path
        )
        logs += "ZIP uploaded.\n"
        yield logs, None
    except Exception as e:
        logs += f"ZIP upload failed: {e}\n"
        yield logs, None
        return

    # --------------------------------------
    # Save uploaded image
    # --------------------------------------

    try:
        img_path = os.path.join(WORK_DIR, "source.jpg")
        shutil.copy(
            image_file.name if hasattr(image_file, "name") else image_file,
            img_path
        )
        logs += "Source image uploaded.\n"
        yield logs, None
    except Exception as e:
        logs += f"Image upload failed: {e}\n"
        yield logs, None
        return

    # --------------------------------------
    # Save user script
    # --------------------------------------

    try:
        script_path = os.path.join(WORK_DIR, "user_script.py")
        with open(script_path, "w") as f:
            f.write(script_text)
        logs += "Script saved.\n"
        yield logs, None
    except Exception as e:
        logs += f"Script save failed: {e}\n"
        yield logs, None
        return

    # --------------------------------------
    # Execute script
    # --------------------------------------

    logs += "Starting script execution...\n"
    yield logs, None

    try:
        process = subprocess.Popen(
            ["python", script_path],
            cwd=WORK_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
    except Exception as e:
        logs += f"Failed to start script: {e}\n"
        yield logs, None
        return

    # Stream logs live
    while True:
        line = process.stdout.readline()
        if not line:
            break
        logs += line
        yield logs, None

    process.wait()

    logs += "\nScript finished.\n"
    yield logs, None

    # --------------------------------------
    # Check for output ZIP
    # --------------------------------------

    output_zip = os.path.join(WORK_DIR, "output_result.zip")

    if os.path.exists(output_zip):
        logs += "Output ZIP found. Ready for download.\n"
        yield logs, output_zip
    else:
        logs += "❌ output_result.zip not found.\n"
        yield logs, None


# ==========================================================
# WEB UI
# ==========================================================

with gr.Blocks() as demo:

    gr.Markdown("## GPU Face Swap Executor")

    gr.Markdown(
        "Upload a ZIP (video inside), upload source image, "
        "paste your face swap script, then run."
    )

    zip_input = gr.File(label="Upload ZIP (video inside)")
    img_input = gr.File(label="Upload Source Image")
    script_input = gr.Textbox(
        lines=20,
        label="Paste Face Swap Script Here"
    )

    run_button = gr.Button("Run Script")

    logs_output = gr.Textbox(
        label="Live Logs",
        lines=20
    )

    file_output = gr.File(
        label="Download output_result.zip"
    )

    run_button.click(
        run_user_script,
        inputs=[zip_input, img_input, script_input],
        outputs=[logs_output, file_output]
    )

# ==========================================================
# LAUNCH SERVER
# ==========================================================

demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)
```

Run app:

```bash
python app.py
```

Expose port **7860** in RunPod networking.

---

# 5️⃣ Full GPU Face Swap Script (Paste Into Web UI)

This script expects:

```
input.zip
source.jpg
```

Produces:

```
output_result.zip
```

```python
import cv2
import os
import zipfile
import glob
import shutil
import insightface
from insightface.app import FaceAnalysis

PROVIDERS = ['CUDAExecutionProvider']
DET_SIZE = (640, 640)

WORK_DIR = os.getcwd()
TEMP_DIR = os.path.join(WORK_DIR, "temp_extract")

# --------------------------------------
# Prepare temp directory
# --------------------------------------

if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

os.makedirs(TEMP_DIR)

print("Extracting zip...")

with zipfile.ZipFile("input.zip", 'r') as zip_ref:
    zip_ref.extractall(TEMP_DIR)

# --------------------------------------
# Find video
# --------------------------------------

video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
video_path = None

for ext in video_extensions:
    files = glob.glob(os.path.join(TEMP_DIR, ext))
    if files:
        video_path = files[0]
        break

if video_path is None:
    raise RuntimeError("No video found in zip")

print("Video:", video_path)

# --------------------------------------
# Load models
# --------------------------------------

print("Loading models...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=PROVIDERS,
    allowed_modules=['detection','recognition']
)

app.prepare(ctx_id=0, det_size=DET_SIZE)

model_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")

swapper = insightface.model_zoo.get_model(
    model_path,
    providers=PROVIDERS
)

if swapper is None:
    raise RuntimeError("Failed to load inswapper")

# --------------------------------------
# Load source face
# --------------------------------------

print("Loading source face...")

src_img = cv2.imread("source.jpg")

faces_src = app.get(src_img)

if not faces_src:
    raise RuntimeError("No face detected in source image")

source_face = faces_src[0]

# --------------------------------------
# Open video
# --------------------------------------

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video = "output.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print("Processing video...")

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)

    if faces:
        frame = swapper.get(frame, faces[0], source_face, paste_back=True)

    out.write(frame)

    frame_count += 1

    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
out.release()

print("Video processing finished")

# --------------------------------------
# Zip result
# --------------------------------------

zip_name = "output_result.zip"

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(output_video)

print("Done:", zip_name)
```

---

# 6️⃣ Performance Notes

RTX A4000:

```
480p → ~60–100 FPS
GPU utilization ~60%
VRAM usage ~2–4GB
```

To increase speed:

* Lower `DET_SIZE` to `(512, 512)`
* Change ffmpeg preset to `veryfast`
* Detect every 2 frames (if face stable)

---

# 7️⃣ GPU Monitoring

```bash
watch -n 1 nvidia-smi
```

---

# 8️⃣ Common Issues We Solved

### Swapper was None

Cause:

```
model not downloaded
```

Fix:

```
manual Dropbox download
```

---

### GPU 0% utilization

Cause:

```
CPU onnxruntime installed
```

Fix:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

---

### Model downloaded but tiny file

Cause:

```
HTML error page saved
```

Fix:

```
correct direct Dropbox link
```

---

# ✅ Final Working Pipeline

```
Upload → Process → GPU Swap → Download ZIP
```

Fully reproducible.

```
```
