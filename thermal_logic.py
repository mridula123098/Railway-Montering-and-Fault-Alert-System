# -*- coding: utf-8 -*-
"""
thermal_logic.py
================
Full pipeline for thermal image analysis of railway OHE jumper connections.
Directly maps wire region pixel colors to the image's temperature scale.
"""

import cv2
import numpy as np
import re
import os
import pandas as pd
from datetime import datetime
from PIL import Image

try:
    import pytesseract
except ImportError:
    raise ImportError("Run: pip install pytesseract")

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


# ═══════════════════════════════════════════════════════════════════
# OCR HELPERS
# ═══════════════════════════════════════════════════════════════════

def parse_scale_val(crop_bgr):
    """OCR scale box and return signed float value (detecting negative signs)."""
    big = cv2.resize(
        crop_bgr,
        (crop_bgr.shape[1] * 8, crop_bgr.shape[0] * 8),
        interpolation=cv2.INTER_LANCZOS4
    )
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    best_val = None
    for thr in [150, 180, 120, 90]:
        _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        for psm in [6, 7, 8, 13]:
            cfg = f"--psm {psm} -c tessedit_char_whitelist=-0123456789."
            txt = pytesseract.image_to_string(Image.fromarray(th), config=cfg).strip()
            
            matches = re.findall(r"(-?\d+\.?\d*)", txt)
            if matches:
                try:
                    val = float(matches[0])
                    if "-" in txt and val > 0:
                        val = -val
                    best_val = val
                    break
                except ValueError:
                    continue
        if best_val is not None:
            break

    return best_val


# ═══════════════════════════════════════════════════════════════════
# LUT COLOR-TO-TEMPERATURE MAPPING
# ═══════════════════════════════════════════════════════════════════

def build_lut(scale, t_max, t_min, n_samples=256):
    """Sample exact RGB colors along the color scale bar and map to temperature range."""
    sh, sw = scale.shape[:2]
    bar_start = int(sh * 0.15)
    bar_end   = int(sh * 0.85)
    bar_strip = scale[bar_start:bar_end, :, :]

    rows = np.linspace(0, bar_strip.shape[0] - 1, n_samples, dtype=int)
    colors = np.array(
        [bar_strip[r].mean(axis=0) for r in rows],
        dtype=np.float32
    )
    temps = np.linspace(t_max, t_min, n_samples, dtype=np.float32)
    return colors, temps


def map_pixels_to_temperature(image_bgr, scale, t_max, t_min):
    """Map every pixel in the image directly to its corresponding temperature via LUT."""
    lut_colors, lut_temps = build_lut(scale, t_max, t_min)
    h, w = image_bgr.shape[:2]
    pixels = image_bgr.reshape(-1, 3).astype(np.float32)

    temp_flat = np.zeros(pixels.shape[0], dtype=np.float32)
    batch_size = 10000

    for i in range(0, pixels.shape[0], batch_size):
        batch = pixels[i:i + batch_size]
        diff = batch[:, None, :] - lut_colors[None, :, :]
        dist = np.sum(diff ** 2, axis=2)
        nearest = np.argmin(dist, axis=1)
        temp_flat[i:i + batch_size] = lut_temps[nearest]

    return temp_flat.reshape(h, w)


# ═══════════════════════════════════════════════════════════════════
# DIRECT WIRE REGION OBSERVATION & ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def segment_wire_and_compute_delta_t(temp_map, t_max_scale, t_min_scale, color_img):
    """Segment the illuminated wire structures and extract true pixel temperature bounds."""
    h, w = temp_map.shape

    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Mask out UI margins
    scene_mask = np.zeros((h, w), dtype=np.uint8)
    scene_mask[int(h * 0.12):int(h * 0.88), int(w * 0.02):int(w * 0.88)] = 1

    scene_gray = gray[scene_mask == 1]
    if scene_gray.size == 0:
        return {
            "wire_t_max": None,
            "wire_t_min": None,
            "delta_t": None,
            "alert": "No wire detected",
            "wire_mask": scene_mask
        }

    p_thresh = float(np.percentile(scene_gray, 92))
    bright_mask = (gray >= p_thresh) & (scene_mask == 1)

    white_core = (S < 90) & (V > 180)
    warm_core = ((H <= 35) | (H >= 160)) & (V > 150)
    wire_candidate = (bright_mask | white_core | warm_core) & (scene_mask == 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(wire_candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Erode to sample strict core pixels only
    wire_core_mask = cv2.erode(cleaned_mask, kernel, iterations=1)
    eval_mask = wire_core_mask if np.count_nonzero(wire_core_mask) > 20 else cleaned_mask

    wire_temps = temp_map[eval_mask == 1]
    wire_temps = wire_temps[np.isfinite(wire_temps)]

    if wire_temps.size == 0:
        return {
            "wire_t_max": None,
            "wire_t_min": None,
            "delta_t": None,
            "alert": "No wire pixels found",
            "wire_mask": cleaned_mask
        }

    # Extract bounds strictly from wire core
    wire_t_max = float(np.percentile(wire_temps, 98.0))
    wire_t_min = float(np.percentile(wire_temps, 25.0))

    if wire_t_max < wire_t_min:
        wire_t_max, wire_t_min = wire_t_min, wire_t_max

    delta_t = wire_t_max - wire_t_min

    if delta_t > 20:
        alert = "CRITICAL - Attend within 24 hrs"
    elif delta_t > 10:
        alert = "WARNING - Attend within 10 days"
    elif delta_t > 5:
        alert = "MONITOR - Attend within 1 month"
    else:
        alert = "NORMAL"

    return {
        "wire_t_max": wire_t_max,
        "wire_t_min": wire_t_min,
        "delta_t": delta_t,
        "alert": alert,
        "wire_mask": cleaned_mask
    }


# ═══════════════════════════════════════════════════════════════════
# STATION LOOKUP
# ═══════════════════════════════════════════════════════════════════

def get_station_from_filename(image_filename, excel_path=None):
    """Match timestamp in image filename to nearest OHE mast/station in reference sheet."""
    try:
        basename = os.path.splitext(os.path.basename(image_filename))[0]
        parts    = basename.split("-")
        if len(parts) < 2:
            return None
        time_str = parts[1]
        img_time = datetime.strptime(time_str, "%H%M%S").time()

        SHEET_ID = "13W4XDKVK384EfZ5rxtccApLsMkca_Jz22qzz-uyrHf8"
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
        df = pd.read_csv(url)

        def find_col(df, keywords):
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in keywords):
                    return col
            return None

        col_section  = find_col(df, ["section", "station", "name"])
        col_ohe      = find_col(df, ["ohe", "mast"])
        col_datetime = find_col(df, ["date", "time", "datetime"])

        if not col_datetime or not col_section:
            return None

        def parse_dt(val):
            s = str(val).strip().replace(" UTC", "")
            for fmt in ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(s, fmt)
                except Exception:
                    continue
            return None

        df["parsed_dt"] = df[col_datetime].apply(parse_dt)
        df = df.dropna(subset=["parsed_dt"])

        if df.empty:
            return None

        def to_secs(t):
            return t.hour * 3600 + t.minute * 60 + t.second

        img_secs        = to_secs(img_time)
        df["diff_secs"] = df["parsed_dt"].apply(
            lambda dt: abs(to_secs(dt.time()) - img_secs)
        )

        nearest = df.loc[df["diff_secs"].idxmin()]

        if nearest["diff_secs"] <= 300:
            ohe_raw = nearest[col_ohe] if col_ohe else "N/A"
            ohe_str = str(ohe_raw).strip().replace(".0", "")

            return {
                "section"      : str(nearest[col_section]).strip(),
                "ohe_mast"     : ohe_str,
                "matched_time" : nearest["parsed_dt"].strftime("%H:%M:%S"),
                "diff_seconds" : int(nearest["diff_secs"])
            }
    except Exception as e:
        print(f"[station lookup error] {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def process_image(image_path):
    color_img = cv2.imread(image_path)
    if color_img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = color_img.shape[:2]

    scale = color_img[:, int(w * 0.90):int(w * 0.98)]
    sh, sw = scale.shape[:2]

    top_crop = scale[int(sh * 0.20):int(sh * 0.28), :]
    bottom_crop = scale[int(sh * 0.78):int(sh * 0.86), :]

    t_max = parse_scale_val(top_crop)
    t_min = parse_scale_val(bottom_crop)

    if t_max is None or t_min is None:
        raise ValueError("Could not OCR temperature scale values from thermal image.")

    if t_max < t_min:
        t_max, t_min = t_min, t_max

    temp_map = map_pixels_to_temperature(color_img, scale, t_max, t_min)
    result = segment_wire_and_compute_delta_t(temp_map, t_max, t_min, color_img)

    return {
        "scale_t_max": t_max,
        "scale_t_min": t_min,
        "max_temp": round(result["wire_t_max"], 1) if result["wire_t_max"] is not None else None,
        "min_temp": round(result["wire_t_min"], 1) if result["wire_t_min"] is not None else None,
        "delta": round(result["delta_t"], 1) if result["delta_t"] is not None else None,
        "status": result["alert"],
        "temp_map": temp_map,
        "wire_mask": result["wire_mask"]
    }
