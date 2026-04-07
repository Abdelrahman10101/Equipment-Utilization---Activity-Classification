# 🦅 Eagle Vision — Kaggle GPU Pipeline

## Quick Start (Step by Step)

### 1. Upload Your Videos
- Go to [Kaggle Datasets](https://www.kaggle.com/datasets) → **New Dataset**
- Upload your video files from `sample_videos/`
- Name it e.g. `eagle-vision-sample-videos`

### 2. Create a New Kaggle Notebook
- Go to [Kaggle Notebooks](https://www.kaggle.com/code) → **New Notebook**
- Click **File → Upload Notebook** and upload `eagle_vision_kaggle.py`
- OR copy-paste the entire file content into a single cell

### 3. Enable GPU
- Click **Settings** (⚙️ gear icon on the right panel)
- Under **Accelerator** → select **GPU T4 x2** or **GPU P100**
- This gives you:
  - 🖥️ **NVIDIA T4** (16GB VRAM) or P100
  - 💾 **~30 GB RAM**
  - ⚡ **4 CPU cores**
  - 📁 **~70 GB disk**

### 4. Add Your Video Dataset
- In the notebook, click **Add Data** (right panel)
- Search for your uploaded dataset
- It will be mounted at `/kaggle/input/your-dataset-name/`

### 5. Update the Path
In the `Config` class, update:
```python
VIDEO_DIR = "/kaggle/input/your-dataset-name"
```

### 6. Run!
Execute the notebook. It will:
1. Auto-detect GPU and load YOLOv8
2. Process each video through the full pipeline
3. Output annotated videos to `/kaggle/working/output_videos/`
4. Save event CSV to `/kaggle/working/equipment_events.csv`
5. Generate utilization reports in `/kaggle/working/reports/`

---

## What's Different from Docker Version?

| Feature | Docker (Original) | Kaggle (This) |
|---|---|---|
| **Messaging** | Kafka topics | Direct function calls |
| **Database** | TimescaleDB | CSV files |
| **Dashboard** | Streamlit (live) | PNG charts + CSV reports |
| **Video I/O** | Kafka streams | Direct file read/write |
| **GPU** | Manual NVIDIA setup | Auto-detected |
| **Infra** | 6+ containers | Single Python script |

**The CV logic is 100% identical** — same detector, same motion analyzer, same activity classifier, same state manager, same annotation renderer.

---

## Output Files

```
/kaggle/working/
├── output_videos/
│   ├── video1_annotated.mp4      # Annotated output video
│   └── video2_annotated.mp4
├── equipment_events.csv          # All per-frame events (like TimescaleDB)
└── reports/
    ├── video1_summary.csv        # Per-equipment utilization summary
    ├── video1_charts.png         # Utilization charts
    ├── video2_summary.csv
    └── video2_charts.png
```

## Using with VSCode Remote (Your Kaggle URL)

Since you have a VSCode-compatible Kaggle URL:

1. Connect VS Code to Kaggle via the Jupyter server URL
2. Open `eagle_vision_kaggle.py` or create a notebook
3. Run cells — GPU is automatically available
4. Download outputs from `/kaggle/working/` when done
