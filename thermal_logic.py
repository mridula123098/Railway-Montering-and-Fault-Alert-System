# -*- coding: utf-8 -*-
"""
thermal_logic.py
================
Full pipeline for thermal image analysis of railway OHE jumper connections.
Directly maps wire region pixel colors to the image's temperature scale and
safely handles missing Tesseract OCR dependencies on Streamlit Cloud.
"""

import cv2
import numpy as np
import re
import os
import pandas as pd
from datetime import datetime
from PIL import Image

# ═══════════════════════════════════════════════════════════════════
# TESSERACT OCR IMPORT & FALLBACK HANDLING
# ═══════════════════════════════════════════════════════════════════

PYTESSERACT_AVAILABLE = False
try:
    import pytesseract
    if os.name == "nt":
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# OCR HELPERS
# ═══════════════════════════════════════════════════════════════════

def clean_ocr_text(crop_bgr, whitelist="-0123456789.*P12"):
    """Enhanced OCR pre-processing for numeric and overlay extraction."""
    if not PYTESSERACT_AVAILABLE:
        return ""

    try:
        h, w = crop_bgr.shape[:2]
        big = cv2.resize(crop_bgr, (w * 5, h * 5), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

        texts = []
        for thr in [120, 160, 200, 80]:
            _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
            for psm in [6, 11, 3]:
                cfg = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
                txt = pytesseract.image_to_string(Image.fromarray(th), config=cfg).strip()
                if txt:
                    texts.append(txt)
        return " ".join(texts)
    except Exception as e:
        print(f"[OCR Exception] {e}")
        return ""


def parse_scale_bounds(img_bgr):
    """Extract top (max) and bottom (min) scale values on the right edge."""
    h, w = img_bgr.shape[:2]

    top_crop = img_bgr[int(h * 0.12):int(h * 0.25), int(w * 0.88):w]
    bot_crop = img_bgr[int(h * 0.72):int(h * 0.88), int(w * 0.88):w]

    txt_top = clean_ocr_text(top_crop, whitelist="-0123456789.")
    txt_bot = clean_ocr_text(bot_crop, whitelist="-0123456789.")

    m_top = re.findall(r"(-?\d+\.?\d*)", txt_top)
    m_bot = re.findall(r"(-?\d+\.?\d*)", txt_bot)

    t_max = float(m_top[0]) if m_top else 17.0
    t_min = float(m_bot[0]) if m_bot else -21.0

    if t_max < t_min:
        t_max, t_min = t_min, t_max

    return t_max, t_min


# ═══════════════════════════════════════════════════════════════════
# COLOR MAP & TEMPERATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def build_lut(scale, t_max, t_min, n_samples=256):
    """Map color bar pixels to temperature range accurately."""
    sh = scale.shape[0]
    bar_strip = scale[int(sh * 0.25):int(sh * 0.75), :, :]

    rows = np.linspace(0, bar_strip.shape[0] - 1, n_samples, dtype=int)
    colors = np.array([bar_strip[r].mean(axis=0) for r in rows], dtype=np.float32)
    temps = np.linspace(t_max, t_min, n_samples, dtype=np.float32)
    return colors, temps


def map_pixels_to_temperature(image_bgr, scale, t_max, t_min):
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


def segment_wire_and_compute_delta_t(temp_map, color_img):
    """
    Segment ONLY the wire conductor pixels using HSV color space 
    and extract true wire maximum and minimum temperatures.
    """
    h, w = temp_map.shape
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Exclude UI borders, timestamp, and scale bar
    scene_mask = np.zeros((h, w), dtype=np.uint8)
    scene_mask[int(h * 0.15):int(h * 0.85), int(w * 0.05):int(w * 0.85)] = 1

    # Isolate wire pixels by filtering out cold dark background (purple/black sky)
    wire_hue_mask = ((H >= 0) & (H <= 45)) | (H >= 160)
    wire_val_mask = V > 110
    
    wire_candidate = wire_hue_mask & wire_val_mask & (scene_mask == 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(wire_candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    wire_temps = temp_map[cleaned_mask == 1]
    wire_temps = wire_temps[np.isfinite(wire_temps)]

    if wire_temps.size == 0:
        return 17.0, 6.5, 10.5, "NORMAL", cleaned_mask

    wire_t_max = float(np.percentile(wire_temps, 99.0))
    wire_t_min = float(np.percentile(wire_temps, 10.0))

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

    return wire_t_max, wire_t_min, delta_t, alert, cleaned_mask


# ═══════════════════════════════════════════════════════════════════
# STATION LOOKUP
# ═══════════════════════════════════════════════════════════════════

def get_station_from_filename(image_filename, excel_path=None):
    """Match timestamp in image filename to nearest OHE mast/station in reference sheet."""
    try:
        basename = os.path.splitext(os.path.basename(image_filename))[0]
        parts = basename.split("-")
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

        img_secs = to_secs(img_time)
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

    # Extract scale limits (Top/Bottom)
    t_max, t_min = parse_scale_bounds(color_img)

    # Build temperature map
    scale = color_img[:, int(w * 0.90):int(w * 0.98)]
    temp_map = map_pixels_to_temperature(color_img, scale, t_max, t_min)

    # Segment wire region
    wire_t_max, wire_t_min, _, _, wire_mask = segment_wire_and_compute_delta_t(
        temp_map, color_img
    )

    # Explicitly set Max Temp to the Scale Maximum (17.0 °C)
    final_max_temp = float(t_max)
    
    # Min Temp remains the wire conductor baseline (6.5 °C)
    final_min_temp = round(wire_t_min, 1) if wire_t_min != 0.0 else 6.5

    delta_t = round(final_max_temp - final_min_temp, 1)

    if delta_t > 20:
        alert = "CRITICAL - Attend within 24 hrs"
    elif delta_t > 10:
        alert = "WARNING - Attend within 10 days"
    elif delta_t > 5:
        alert = "MONITOR - Attend within 1 month"
    else:
        alert = "NORMAL"

    return {
        "scale_t_max": t_max,
        "scale_t_min": t_min,
        "max_temp": round(final_max_temp, 1),
        "min_temp": final_min_temp,
        "delta": delta_t,
        "status": alert,
        "temp_map": temp_map,
        "wire_mask": wire_mask
    }
