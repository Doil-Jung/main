## Thermometer Reader via Webcam + GPT

Captures a webcam photo of a thermometer, reads the temperature using OpenAI's Vision models, and logs results to CSV.

### 1) Install

Requirements: Python 3.9+.

```
pip install -r /workspace/requirements.txt
```

If you plan to use environment variables from a file, optionally create `/workspace/.env`.

### 2) Configure API key

Set your OpenAI API key:

```
export OPENAI_API_KEY=your_key_here
```

Optionally, add to `/workspace/.env`:

```
OPENAI_API_KEY=your_key_here
```

### 3) Usage

Single capture (default webcam at `/dev/video0`):

```
python /workspace/thermo_cam.py --device 0
```

Useful options:

- `--device`: webcam index (0 -> `/dev/video0`)
- `--width` / `--height`: requested resolution
- `--output-dir`: where to save images (default: `/workspace/captures`)
- `--csv`: log file (default: `/workspace/temperature_log.csv`)
- `--model`: OpenAI model (default: `gpt-4o-mini`)

Periodic capture every N seconds:

```
python /workspace/thermo_cam.py --interval 60
```

Force a single run even if `--interval` is given:

```
python /workspace/thermo_cam.py --interval 60 --once
```

### Notes

- The script prefers OpenCV; if that fails and `ffmpeg` is available, it tries a fallback via `/dev/video*`.
- The model response is constrained to JSON, and we log `timestamp`, `image_path`, `temperature_value`, `unit`, and `raw` to the CSV.

