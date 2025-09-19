# Siamese-Neural-Networks-in-C-


This project is a **C++ implementation** of the **Siamese Transformer Pyramid Network (SiamTPN)** for single‑object tracking, converted from a Python baseline and designed to run with **ONNX Runtime**.  
Tested on **Linux Mint** (CPU-oriented).

> **Important:** Place both ONNX model files — `backbone_fpn_z.onnx` and `backbone_fpn_head_x.onnx` — in the **project root folder** (same folder where you run the app or pass relative paths from).

---

## Background 
SiamTPN combines **lightweight CNN backbones** (e.g., ShuffleNetV2) with a **Transformer Pyramid Network (TPN)** that fuses multi‑scale features efficiently. A **Pooling Attention** design keeps attention costs low, enabling **real‑time CPU** tracking (≈30+ FPS reported in the paper), while maintaining competitive accuracy on benchmarks like LaSOT and UAV123.

---

## Dependencies (Linux Mint)
Install build tools and OpenCV:
```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev
```

Download ONNX Runtime (prebuilt binaries), set `ORT_HOME`:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-1.19.0.tgz
tar -xzf onnxruntime-linux-x64-1.19.0.tgz -C $HOME/dev
export ORT_HOME="$HOME/dev/onnxruntime-linux-x64-1.19.0"
```

> You can put the `export ORT_HOME=...` line into your shell profile to persist it across sessions.

---

## Build (CMake)
From the project root (where `tracking.cpp` and the models live):
```bash
mkdir build
cd build
cmake .. -DORT_HOME="$ORT_HOME"
make -j"$(nproc)"
```
This produces an executable named `tracker` in `build/`.

If CMake can’t find OpenCV: make sure `libopencv-dev` is installed and `pkg-config --modversion opencv4` works.

---

## Run
Example command (relative paths assume models/video are in the project root or one level up as shown):
```bash
./tracker   --video ../test.mp4   --z ../backbone_fpn_z.onnx   --x ../backbone_fpn_head_x.onnx   --box-normalized 1   --hanning 0.01   --smooth 0.6   --template-factor 1.5   --show   --save ../tracked_fixed.avi
```

### Arguments
- `--video` : Path to input video file  
- `--z` : Template ONNX model (kernel branch)  
- `--x` : Search/head ONNX model  
- `--box-normalized` : `1` if model outputs are in normalized crop coords; `0` for pixel‑space anchors  
- `--hanning` : Hanning window influence (e.g., `0.01`)  
- `--smooth` : Exponential moving average factor for box smoothing (e.g., `0.6`)  
- `--template-factor` : Template crop enlargement factor (e.g., `1.5`)  
- `--show` : Show live window  
- `--save` : Save annotated video to the given path

On the first frame you’ll be asked to **select an ROI**; press **ENTER/SPACE** to confirm, **ESC** to cancel.

---

## Notes
- The code runs on **CPU** by default. If your ONNX Runtime build supports CUDA, you can enable the CUDA EP by adding the corresponding line in the code where session options are created.  
- FPS is printed overlayed on the output frames.  
- Make sure both ONNX files are present in the **root folder** (or adjust the `--z` / `--x` paths accordingly).
