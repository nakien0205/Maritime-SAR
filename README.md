# Maritime-SAR

YOLO-based maritime search-and-rescue tracking project (SeaDroneSee-style dataset layout).

## What is in this repo

- Trained model weights: `best.pt` (PyTorch), `best.tflite` (TensorFlow Lite)
- Main inference script on image frames: `inference.py`
- MP4 tracking demo script: `real_time_mp4.py`
- Dataset config files: `data/data.yaml`, `data_sahi.yaml`
- Dataset folders under `data/` (`train`, `val`, optional `test`)

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Download dataset from [Kaggle](https://www.kaggle.com/datasets/ubiratanfilho/sds-dataset)

### Dataset Structure

Yes, your dataset should be under the `data/` folder.

At minimum, the notebook expects this layout:

```text
data/
├── train/
│   ├── images/   # training images
│   └── labels/   # YOLO .txt labels for train images
├── val/
│   ├── images/   # validation images
│   └── labels/   # YOLO .txt labels for val images
├── test/
│   ├── images/   # test images
│   └── labels/   # optional for this notebook pipeline
└── annotations/  # optional metadata/json exports
```

## Setup venv

From the project root:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Inference on Frame Sequence (recommended)

The `inference.py` code is for processing each image from cut from a video. This script is parameterized and easiest to use.

```bash
python inference.py --model best.pt --frame-dir data/test --start 6701 --end 6881 --imgsz 640 --conf 0.25 --iou 0.5
```

Notes:
- Press `q` to quit the OpenCV display window.
- If your test frames are in another folder, change `--frame-dir`.
- If your frame names are not numeric (`6701.jpg`, etc.), adjust the script logic in `inference.py`.

## Run Tracking on an MP4 Video

This code `real_time_mp4.py` can process mp4 video and currently uses hardcoded absolute paths. Before running, edit:

- model path in `real_time_mp4.py`
- video path in `real_time_mp4.py`

Then run:

```bash
python real_time_mp4.py
```