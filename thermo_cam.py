import argparse
import base64
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    cv2 = None  # Will try ffmpeg fallback if available

from typing import Any, Dict, Optional, Tuple


def ensure_directory_exists(directory_path: Path) -> None:
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)


def capture_with_opencv(
    device_index: int,
    frame_width: int,
    frame_height: int,
    warmup_frames: int,
    timeout_seconds: int,
    destination_path: Path,
) -> bool:
    if cv2 is None:
        return False

    # Try opening the camera
    capture = cv2.VideoCapture(device_index)

    if not capture.isOpened():
        # Some Linux setups prefer V4L2 explicitly
        capture.open(device_index, cv2.CAP_V4L2)

    if not capture.isOpened():
        return False

    # Configure resolution if supported
    if frame_width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    if frame_height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Warm up
    start_time = time.time()
    for _ in range(max(0, warmup_frames)):
        ok, _ = capture.read()
        if not ok and (time.time() - start_time) > timeout_seconds:
            capture.release()
            return False

    # Capture a single frame
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return False

    # Write JPEG
    ok = cv2.imwrite(str(destination_path), frame)
    return bool(ok)


def capture_with_ffmpeg(device_index: int, destination_path: Path, timeout_seconds: int) -> bool:
    """Attempt capture via ffmpeg as a fallback.

    Requires ffmpeg installed and a Linux video device like /dev/video0.
    """
    device_path = f"/dev/video{device_index}"
    if not Path(device_path).exists():
        return False

    # Single frame capture using ffmpeg; -frames:v 1 grabs one frame
    # We silence the output and enforce a timeout via bash `timeout` if available.
    ffmpeg_cmd = (
        f"bash -lc 'command -v ffmpeg >/dev/null 2>&1 && "
        f"timeout {timeout_seconds}s ffmpeg -hide_banner -loglevel error -f video4linux2 -i {device_path} -frames:v 1 "
        f"-y {str(destination_path)}'"
    )

    try:
        import subprocess

        result = subprocess.run(
            ffmpeg_cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0 and destination_path.exists()
    except Exception:
        return False


def capture_image(
    device_index: int,
    frame_width: int,
    frame_height: int,
    warmup_frames: int,
    timeout_seconds: int,
    output_directory: Path,
    filename_prefix: str = "capture",
) -> Path:
    ensure_directory_exists(output_directory)
    timestamp_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    destination_path = output_directory / f"{filename_prefix}_{timestamp_str}.jpg"

    # Try OpenCV first
    if capture_with_opencv(
        device_index=device_index,
        frame_width=frame_width,
        frame_height=frame_height,
        warmup_frames=warmup_frames,
        timeout_seconds=timeout_seconds,
        destination_path=destination_path,
    ):
        return destination_path

    # Fallback to ffmpeg
    if capture_with_ffmpeg(
        device_index=device_index, destination_path=destination_path, timeout_seconds=timeout_seconds
    ):
        return destination_path

    raise RuntimeError("Failed to capture image from webcam via OpenCV and ffmpeg fallback.")


def image_file_to_data_url(image_path: Path) -> str:
    with image_path.open("rb") as f:
        image_bytes = f.read()
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_data}"


def parse_temperature_json(text: str) -> Dict[str, Any]:
    """Parse JSON returned by the model. Provide robust fallback if needed."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # As a fallback, try to extract a floating number and optional unit from free text
    import re

    match = re.search(r"(-?\d+(?:[.,]\d+)?)\s*([CFcf])?", text)
    temperature_value: Optional[float] = None
    unit: Optional[str] = None
    if match:
        number_str = match.group(1).replace(",", ".")
        try:
            temperature_value = float(number_str)
        except Exception:
            temperature_value = None
        if match.group(2):
            unit = match.group(2).upper()
    return {
        "temperature_value": temperature_value,
        "unit": unit,
        "raw": text.strip(),
    }


def extract_temperature_with_gpt(
    image_data_url: str,
    model: str,
) -> Dict[str, Any]:
    # Defer import so the script can run partial features without the SDK
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "openai SDK is not installed. Please install dependencies from requirements.txt"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or place it in a .env file."
        )

    client = OpenAI(api_key=api_key)

    system_message = (
        "You are a precise OCR + parser that reads the temperature from a photographed thermometer. "
        "Only return strict JSON with keys temperature_value (number|null), unit (\"C\"|\"F\"|null), and raw (string)."
    )
    user_instruction = (
        "Read the temperature shown on this thermometer. "
        "Return JSON only, no extra text, matching: {\"temperature_value\": number|null, \"unit\": \"C\"|\"F\"|null, \"raw\": string}."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instruction},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )

    text = response.choices[0].message.content or "{}"
    return parse_temperature_json(text)


def write_csv_row(
    csv_path: Path,
    timestamp_iso: str,
    image_path: Path,
    temperature_value: Optional[float],
    unit: Optional[str],
    raw: Optional[str],
) -> None:
    is_new_file = not csv_path.exists()
    ensure_directory_exists(csv_path.parent)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp", "image_path", "temperature_value", "unit", "raw"]) 
        writer.writerow([timestamp_iso, str(image_path), temperature_value, unit, (raw or "").strip()])


def run_once(args: argparse.Namespace) -> Tuple[Path, Dict[str, Any]]:
    image_path = capture_image(
        device_index=args.device,
        frame_width=args.width,
        frame_height=args.height,
        warmup_frames=args.warmup,
        timeout_seconds=args.timeout,
        output_directory=Path(args.output_dir),
        filename_prefix="thermo",
    )
    data_url = image_file_to_data_url(image_path)
    result = extract_temperature_with_gpt(data_url, model=args.model)
    timestamp_iso = dt.datetime.now().isoformat(timespec="seconds")
    write_csv_row(
        csv_path=Path(args.csv),
        timestamp_iso=timestamp_iso,
        image_path=image_path,
        temperature_value=result.get("temperature_value"),
        unit=result.get("unit"),
        raw=result.get("raw"),
    )
    return image_path, result


def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        # dotenv is optional
        pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a webcam image of a thermometer, read temperature with GPT, and log to CSV.",
    )
    parser.add_argument("--device", type=int, default=0, help="Webcam device index, e.g., 0 for /dev/video0")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width (if supported)")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height (if supported)")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup frames before capture")
    parser.add_argument("--timeout", type=int, default=5, help="Timeout seconds for capture")
    parser.add_argument("--output-dir", type=str, default=str(Path("/workspace") / "captures"), help="Directory to save images")
    parser.add_argument("--csv", type=str, default=str(Path("/workspace") / "temperature_log.csv"), help="CSV file path for logs")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use for vision parsing")
    parser.add_argument("--interval", type=int, default=0, help="If > 0, seconds between periodic captures")
    parser.add_argument("--once", action="store_true", help="Force single run even if --interval is set")
    return parser


def main(argv: Optional[list] = None) -> int:
    load_dotenv_if_present()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.interval > 0 and not args.once:
        print(
            f"Starting periodic capture every {args.interval} seconds. Press Ctrl+C to stop.",
            flush=True,
        )
        try:
            while True:
                image_path, result = run_once(args)
                print(
                    json.dumps(
                        {
                            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                            "image_path": str(image_path),
                            "result": result,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Stopped.")
            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    else:
        try:
            image_path, result = run_once(args)
            print(json.dumps({"image_path": str(image_path), "result": result}, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())

