# -*- coding: utf-8 -*-
"""
thermal_logic.py
================
Full pipeline for thermal image analysis of railway OHE jumper connections.
Place in the same folder as app.py and station_log.xlsx.
"""

import cv2
import numpy as np
import re
import os
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image

try:
    import pytesseract
except ImportError:
    raise ImportError("Run: pip install pytesseract")

# Windows only — remove this line on Linux / Streamlit Cloud
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


# ═══════════════════════════════════════════════════════════════════
# OCR HELPERS
# ═══════════════════════════════════════════════════════════════════

def crop_to_temp(crop_bgr):
    """OCR a grey label box and return the numeric value (absolute)."""
    big = cv2.resize(
        crop_bgr,
        (crop_bgr.shape[1] * 8, crop_bgr.shape[0] * 8),
        interpolation=cv2.INTER_LANCZOS4
    )
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    best = None
    for psm in [7, 8, 13]:
        cfg = f"--psm {psm} -c tessedit_char_whitelist=0123456789."
        txt = pytesseract.image_to_string(Image.fromarray(th), config=cfg).strip()
        nums = re.findall(r"\d+\.?\d*", txt)
        if nums and best is None:
            best = float(nums[0])
    return best


def has_minus_sign(crop_bgr):
    """Detect a minus sign above/below the grey number box."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    bright_rows = [r for r in range(gray.shape[0]) if gray[r].mean() > 100]
    if not bright_rows:
        return False

    box_start = bright_rows[0]
    box_end   = bright_rows[-1]
    above     = gray[:box_start, :]
    below     = gray[box_end + 1:, :]

    def minus_in(region):
        if region.shape[0] == 0:
            return False
        for row in range(region.shape[0]):
            if region[row].mean() < 80 and region[row].max() > 50:
                return True
        return False

    return minus_in(above) or minus_in(below)


# ═══════════════════════════════════════════════════════════════════
# LUT BUILDING
# ═══════════════════════════════════════════════════════════════════

def build_lut(scale, t_max, t_min, n_samples=256):
    """Sample colors along the color bar and map to temperatures."""
    sh, sw   = scale.shape[:2]
    bar_start = int(sh * 0.25)
    bar_end   = int(sh * 0.75)
    bar_strip = scale[bar_start:bar_end, :, :]

    rows   = np.linspace(0, bar_strip.shape[0] - 1, n_samples, dtype=int)
    colors = np.array(
        [bar_strip[r].mean(axis=0) for r in rows],
        dtype=np.float32
    )
    temps = np.linspace(t_max, t_min, n_samples, dtype=np.float32)
    return colors, temps


def map_pixels_to_temperature(image_bgr, scale, t_max, t_min):
    """Map every pixel in the image to a temperature value via the LUT."""
    lut_colors, lut_temps = build_lut(scale, t_max, t_min)
    h, w   = image_bgr.shape[:2]
    pixels = image_bgr.reshape(-1, 3).astype(np.float32)

    temp_flat  = np.zeros(pixels.shape[0], dtype=np.float32)
    batch_size = 10000

    for i in range(0, pixels.shape[0], batch_size):
        batch   = pixels[i:i + batch_size]
        diff    = batch[:, None, :] - lut_colors[None, :, :]
        dist    = np.sum(diff ** 2, axis=2)
        nearest = np.argmin(dist, axis=1)
        temp_flat[i:i + batch_size] = lut_temps[nearest]

    return temp_flat.reshape(h, w)


# ═══════════════════════════════════════════════════════════════════
# WIRE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════

def segment_wire_and_compute_delta_t(temp_map, t_max_scale, t_min_scale, color_img):
    """Segment the wire region and compute T_max, T_min, Delta T."""
    h, w = temp_map.shape
    mid_thresh = (t_max_scale + t_min_scale) / 2

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[40:h - 40, 160:w - 80] = 1

    above_mid = (
        (temp_map >= mid_thresh) & (roi_mask == 1)
    ).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        above_mid, connectivity=8
    )

    wire_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        bh   = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 30:
            continue
        aspect   = max(bw, bh) / max(min(bw, bh), 1)
        solidity = area / max(bw * bh, 1)
        is_ui    = solidity > 0.50
        is_blob  = area > 1500 and solidity > 0.45 and aspect < 3.0
        is_wire  = (aspect >= 3.0 or solidity < 0.35) and area > 30
        if is_wire and not is_ui and not is_blob:
            wire_mask[labels == i] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_OPEN,  kernel)
    wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, kernel)

    if wire_mask.sum() == 0:
        return {
            "wire_t_max": None,
            "wire_t_min": None,
            "delta_t"   : None,
            "alert"     : "No wire detected",
            "wire_mask" : wire_mask
        }

    wire_temps = temp_map[wire_mask == 1]
    wire_t_max = float(np.percentile(wire_temps, 99))
    wire_t_min = float(np.percentile(wire_temps, 5))
    delta_t    = wire_t_max - wire_t_min

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
        "delta_t"   : delta_t,
        "alert"     : alert,
        "wire_mask" : wire_mask
    }


# ═══════════════════════════════════════════════════════════════════
# STATION LOOKUP
# ═══════════════════════════════════════════════════════════════════

def get_station_from_filename(image_filename, excel_path="station_log.xlsx"):
    try:
        basename = os.path.splitext(os.path.basename(image_filename))[0]
        parts    = basename.split("-")
        if len(parts) < 2:
            return None

        time_str = parts[1]
        img_time = datetime.strptime(time_str, "%H%M%S").time()

        # ── Read Excel with its own header ────────────────────
        # df = pd.read_excel(excel_path, header=0)
        df = pd.read_excel(excel_path)
        
        def find_col(df, keywords):
            """Find column whose name contains any of the keywords."""
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in keywords):
                    return col
            return None
            
        ohe_col = find_col(df, ["ohe", "mast"])
        if ohe_col:
            df[ohe_col] = df[ohe_col].astype(str)

        # Print columns so you can verify (remove after testing)
        print(f"[Excel columns] {df.columns.tolist()}")

        # ── Auto-detect column names (case-insensitive) ───────
        # def find_col(df, keywords):
        #     """Find column whose name contains any of the keywords."""
        #     for col in df.columns:
        #         col_lower = str(col).lower()
        #         if any(kw in col_lower for kw in keywords):
        #             return col
        #     return None

        col_section  = find_col(df, ["section", "station", "name"])
        col_ohe      = find_col(df, ["ohe", "mast"])
        col_datetime = find_col(df, ["date", "time", "datetime"])

        if not col_datetime:
            print("[station lookup] Could not find Date/Time column")
            return None

        if not col_section:
            print("[station lookup] Could not find Section column")
            return None

        # ── Parse dates ───────────────────────────────────────
        def parse_dt(val):
            s = str(val).strip().replace(" UTC", "")
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%d/%m/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M:%S",
                "%Y-%m-%d",
            ]:
                try:
                    return datetime.strptime(s, fmt)
                except Exception:
                    continue
            return None

        df["parsed_dt"] = df[col_datetime].apply(parse_dt)
        df = df.dropna(subset=["parsed_dt"])

        if df.empty:
            print("[station lookup] No rows after date parsing")
            return None

        # ── Match by time only ────────────────────────────────
        def to_secs(t):
            return t.hour * 3600 + t.minute * 60 + t.second

        img_secs        = to_secs(img_time)
        df["diff_secs"] = df["parsed_dt"].apply(
            lambda dt: abs(to_secs(dt.time()) - img_secs)
        )

        nearest = df.loc[df["diff_secs"].idxmin()]

        if nearest["diff_secs"] <= 10:
            print("DEBUG: MATCH FOUND")
            return {
                "section"      : str(nearest[col_section]).strip(),
                "ohe_mast"     : str(nearest[col_ohe]).strip() if col_ohe else "N/A",
                # "ohe_mast": str(row[ohe_col]).split(" ")[0],
                "matched_time" : nearest["parsed_dt"].strftime("%H:%M:%S"),
                "diff_seconds" : int(nearest["diff_secs"])
            }
        print("DEBUG: returning None because diff_secs > 10") 
        return None

    except Exception as e:
        print(f"[station lookup error] {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def process_image(image_path):
    """
    Full pipeline: load → extract scale → build temp map → segment wire → alert.

    Returns dict:
        scale_t_max, scale_t_min, max_temp, min_temp, delta, status,
        temp_map, wire_mask
    """
    color_img = cv2.imread(image_path)
    if color_img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w  = color_img.shape[:2]

    # ── Scale strip (rightmost ~4% of width) ─────────────────────
    scale = color_img[:, int(w * 0.94):int(w * 0.98)]
    sh, sw = scale.shape[:2]

    top    = scale[int(sh * 0.13):int(sh * 0.23), :]
    bottom = scale[int(sh * 0.78):int(sh * 0.88), :]

    # ── OCR temperatures ─────────────────────────────────────────
    t_max_abs = crop_to_temp(top)
    t_min_abs = crop_to_temp(bottom)

    top_is_negative = has_minus_sign(top)
    bot_is_negative = has_minus_sign(bottom)

    t_max = -t_max_abs if (top_is_negative and t_max_abs) else t_max_abs
    t_min = -t_min_abs if (bot_is_negative and t_min_abs) else t_min_abs

    # ── Sanity check: top must be hotter than bottom ──────────────
    if t_max is not None and t_min is not None and t_max < t_min:
        if top_is_negative and not bot_is_negative:
            t_max = t_max_abs
        elif bot_is_negative and not top_is_negative:
            t_min = t_min_abs
        else:
            t_max = max(t_max_abs or 0, t_min_abs or 0)
            t_min = min(t_max_abs or 0, t_min_abs or 0)

    # ── Temperature map ───────────────────────────────────────────
    temp_map = map_pixels_to_temperature(color_img, scale, t_max, t_min)

    # ── Wire segmentation + Delta T ───────────────────────────────
    result = segment_wire_and_compute_delta_t(
        temp_map, t_max, t_min, color_img
    )

    return {
        "scale_t_max": t_max,
        "scale_t_min": t_min,
        "max_temp"   : result["wire_t_max"],
        "min_temp"   : result["wire_t_min"],
        "delta"      : result["delta_t"],
        "status"     : result["alert"],
        "temp_map"   : temp_map,
        "wire_mask"  : result["wire_mask"]
    }
