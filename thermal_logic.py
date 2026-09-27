# -*- coding: utf-8 -*-
"""
thermal_logic.py
================

Automatic thermal image analysis for railway OHE jumper / junction
connections.

import cv2
import numpy as np
import re
import os
import pandas as pd
from datetime import datetime
from PIL import Image


# ═══════════════════════════════════════════════════════════════════
# TESSERACT
# ═══════════════════════════════════════════════════════════════════

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

def has_minus_sign(text):
    """
    Check whether OCR text contains a negative sign.

    This is kept separate because OCR can sometimes recognise:

        -28

    as:

        28

    The minus sign must therefore be preserved independently.
    """

    if text is None:
        return False

    text = str(text)

    return bool(
        re.search(
            r"[-−]",
            text
        )
    )


def parse_scale_val(crop_bgr):
    """
    OCR one temperature-scale label.

    Returns:
        signed float
        or None if OCR fails.

    Negative values are preserved using has_minus_sign().
    """

    if crop_bgr is None or crop_bgr.size == 0:
        return None

    big = cv2.resize(
        crop_bgr,
        (
            crop_bgr.shape[1] * 8,
            crop_bgr.shape[0] * 8
        ),
        interpolation=cv2.INTER_LANCZOS4
    )

    gray = cv2.cvtColor(
        big,
        cv2.COLOR_BGR2GRAY
    )

    best_value = None
    best_text = ""

    thresholds = [
        150,
        180,
        120,
        90
    ]

    psm_modes = [
        6,
        7,
        8,
        13
    ]

    for threshold in thresholds:

        _, binary = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

        for psm in psm_modes:

            config = (
                f"--psm {psm} "
                "-c tessedit_char_whitelist=-−0123456789."
            )

            text = pytesseract.image_to_string(
                Image.fromarray(binary),
                config=config
            ).strip()

            if not text:
                continue

            # Preserve the original OCR text so that the minus sign
            # can be checked independently.
            if has_minus_sign(text):
                negative = True
            else:
                negative = False

            matches = re.findall(
                r"-?\d+(?:\.\d+)?",
                text
            )

            if not matches:
                # OCR may separate '-' and number.
                matches = re.findall(
                    r"\d+(?:\.\d+)?",
                    text
                )

            if not matches:
                continue

            try:

                value = float(
                    matches[0].replace("−", "-")
                )

                if negative:
                    value = -abs(value)

                best_value = value
                best_text = text

                break

            except ValueError:
                continue

        if best_value is not None:
            break

    return best_value


# ═══════════════════════════════════════════════════════════════════
# THERMAL LUT
# ═══════════════════════════════════════════════════════════════════

def build_lut(
    scale,
    t_max,
    t_min,
    n_samples=256
):
    """
    Build colour → temperature lookup table from the thermal scale.

    The temperature range is taken from the detected image scale.
    No fixed temperature range is assumed.
    """

    if scale is None or scale.size == 0:
        raise ValueError(
            "Thermal scale image is empty."
        )

    sh, sw = scale.shape[:2]

    # Ignore the white numeric labels at the top/bottom
    # and focus on the actual colour bar.
    bar_start = int(sh * 0.15)
    bar_end = int(sh * 0.85)

    bar_strip = scale[
        bar_start:bar_end,
        :,
        :
    ]

    if bar_strip.size == 0:
        raise ValueError(
            "Could not extract thermal colour bar."
        )

    rows = np.linspace(
        0,
        bar_strip.shape[0] - 1,
        n_samples,
        dtype=int
    )

    colors = np.array(
        [
            bar_strip[row].mean(axis=0)
            for row in rows
        ],
        dtype=np.float32
    )

    temps = np.linspace(
        float(t_max),
        float(t_min),
        n_samples,
        dtype=np.float32
    )

    return colors, temps


def map_pixels_to_temperature(
    image_bgr,
    scale,
    t_max,
    t_min
):
    """
    Convert every thermal-image pixel into an estimated temperature
    using the detected colour scale.
    """

    lut_colors, lut_temps = build_lut(
        scale,
        t_max,
        t_min
    )

    h, w = image_bgr.shape[:2]

    pixels = image_bgr.reshape(
        -1,
        3
    ).astype(
        np.float32
    )

    temp_flat = np.zeros(
        pixels.shape[0],
        dtype=np.float32
    )

    batch_size = 10000

    for start in range(
        0,
        pixels.shape[0],
        batch_size
    ):

        end = min(
            start + batch_size,
            pixels.shape[0]
        )

        batch = pixels[
            start:end
        ]

        diff = (
            batch[:, None, :]
            -
            lut_colors[None, :, :]
        )

        distance = np.sum(
            diff ** 2,
            axis=2
        )

        nearest = np.argmin(
            distance,
            axis=1
        )

        temp_flat[
            start:end
        ] = lut_temps[
            nearest
        ]

    return temp_flat.reshape(
        h,
        w
    )


# ═══════════════════════════════════════════════════════════════════
# AUTOMATIC OHE WIRE / JUNCTION DETECTION
# ═══════════════════════════════════════════════════════════════════

def create_ohe_candidate_mask(color_img):
    """
    Automatically identify bright thermal OHE structures.

    This does NOT use an absolute temperature threshold.

    The algorithm uses:
        - adaptive brightness
        - HSV colour information
        - spatial exclusion
        - morphology
        - connected components

    It also attempts to remove:
        - thermal scale
        - OCR text
        - timestamp
        - image borders
        - dark/purple background
    """

    h, w = color_img.shape[:2]

    hsv = cv2.cvtColor(
        color_img,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        color_img,
        cv2.COLOR_BGR2GRAY
    )

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # ---------------------------------------------------------------
    # SPATIAL ANALYSIS AREA
    # ---------------------------------------------------------------
    #
    # Based on the supplied thermal-image format:
    #
    # - upper area contains camera information / P1 / P2 text
    # - right side contains thermal colour scale
    # - bottom area may contain timestamp
    #
    # The actual OHE structures occupy the central scene.
    #
    # These are spatial exclusions, NOT temperature thresholds.
    # ---------------------------------------------------------------

    scene_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    y_start = int(h * 0.30)
    y_end = int(h * 0.86)

    x_start = int(w * 0.02)
    x_end = int(w * 0.90)

    scene_mask[
        y_start:y_end,
        x_start:x_end
    ] = 1

    scene_gray = gray[
        scene_mask == 1
    ]

    if scene_gray.size == 0:
        return np.zeros(
            (h, w),
            dtype=np.uint8
        )

    # ---------------------------------------------------------------
    # ADAPTIVE BRIGHTNESS
    # ---------------------------------------------------------------

    brightness_threshold = float(
        np.percentile(
            scene_gray,
            88
        )
    )

    bright_mask = (
        gray >= brightness_threshold
    )

    # ---------------------------------------------------------------
    # VERY BRIGHT / LOW-SATURATION THERMAL PIXELS
    #
    # White-hot / bright wire pixels generally have:
    #     high V
    #     lower S
    # ---------------------------------------------------------------

    white_thermal = (
        (V >= brightness_threshold) &
        (S <= 120)
    )

    # ---------------------------------------------------------------
    # WARM THERMAL PIXELS
    #
    # Yellow / orange / red portions of the thermal wire.
    #
    # This is based on colour family, not temperature.
    # ---------------------------------------------------------------

    warm_thermal = (
        (
            (H <= 38) |
            (H >= 170)
        ) &
        (S >= 70) &
        (V >= brightness_threshold)
    )

    # ---------------------------------------------------------------
    # COMBINE
    # ---------------------------------------------------------------

    candidate = (
        (
            white_thermal |
            warm_thermal
        ) &
        (bright_mask | white_thermal) &
        (scene_mask == 1)
    )

    candidate = candidate.astype(
        np.uint8
    )

    # ---------------------------------------------------------------
    # MORPHOLOGICAL CLEANUP
    # ---------------------------------------------------------------

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        small_kernel
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        small_kernel
    )

    return candidate


def select_ohe_components(candidate_mask):
    """
    Select meaningful OHE-like connected components.

    Preference is given to:
        - elongated structures
        - sufficiently large structures
        - structures spanning a useful portion of the image

    Returns a binary wire/junction mask.
    """

    h, w = candidate_mask.shape

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            candidate_mask,
            connectivity=8
        )
    )

    components = []

    for label in range(
        1,
        num_labels
    ):

        x = stats[
            label,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            label,
            cv2.CC_STAT_TOP
        ]

        ww = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        hh = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area < 40:
            continue

        min_dim = max(
            min(ww, hh),
            1
        )

        aspect_ratio = (
            max(ww, hh)
            /
            min_dim
        )

        width_ratio = ww / max(
            w,
            1
        )

        height_ratio = hh / max(
            h,
            1
        )

        # OHE wires are normally elongated.
        elongated = (
            aspect_ratio >= 2.0
        )

        # Prefer structures that span a meaningful width,
        # since the actual OHE wires generally extend across
        # a substantial part of the thermal image.
        spans_scene = (
            width_ratio >= 0.15
        )

        # Junctions can be compact, so don't require every
        # component to be extremely elongated.
        meaningful = (
            area >= 80
        )

        if meaningful and (
            elongated or spans_scene
        ):

            components.append(
                {
                    "label": label,
                    "area": area,
                    "aspect": aspect_ratio,
                    "width_ratio": width_ratio,
                    "height_ratio": height_ratio
                }
            )

    output_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    if not components:
        return output_mask

    # ---------------------------------------------------------------
    # SCORE COMPONENTS
    # ---------------------------------------------------------------
    #
    # We want actual OHE structures rather than isolated noise.
    # ---------------------------------------------------------------

    for component in components:

        label = component["label"]
        area = component["area"]
        aspect = component["aspect"]
        width_ratio = component["width_ratio"]

        score = (
            area
            *
            (
                1.0
                +
                min(aspect, 15.0) * 0.15
                +
                width_ratio * 2.0
            )
        )

        component["score"] = score

    components.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Keep the strongest connected OHE structures.
    #
    # More than one component can belong to the same OHE
    # junction assembly.
    selected = components[
        :min(8, len(components))
    ]

    for component in selected:

        output_mask[
            labels == component["label"]
        ] = 1

    # ---------------------------------------------------------------
    # CONNECT NEARBY WIRE PARTS
    # ---------------------------------------------------------------

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    output_mask = cv2.morphologyEx(
        output_mask,
        cv2.MORPH_CLOSE,
        connect_kernel
    )

    return output_mask


def refine_ohe_mask(
    wire_mask,
    color_img
):
    """
    Refine automatically detected OHE mask.

    Removes obvious low-brightness/background pixels while retaining
    bright thermal structures.
    """

    hsv = cv2.cvtColor(
        color_img,
        cv2.COLOR_BGR2HSV
    )

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Calculate brightness only inside the candidate region.
    candidate_values = V[
        wire_mask == 1
    ]

    if candidate_values.size == 0:
        return wire_mask

    # Adaptive threshold based on detected candidate pixels.
    v_threshold = float(
        np.percentile(
            candidate_values,
            20
        )
    )

    # Keep pixels sufficiently bright relative to the detected
    # OHE structure.
    brightness_keep = (
        V >= v_threshold
    )

    # Purple/blue background generally has a different hue range.
    # This is intentionally not used as the only detector.
    blue_background = (
        (H >= 95) &
        (H <= 145) &
        (S >= 70) &
        (V < v_threshold)
    )

    refined = (
        (wire_mask == 1) &
        brightness_keep &
        (~blue_background)
    ).astype(
        np.uint8
    )

    # Small cleanup.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_OPEN,
        kernel
    )

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        kernel
    )

    return refined


# ═══════════════════════════════════════════════════════════════════
# TEMPERATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def segment_wire_and_compute_delta_t(
    temp_map,
    t_max_scale,
    t_min_scale,
    color_img
):
    """
    Automatically detect OHE wire/junction and calculate temperature.

    No manual ROI is required.
    """

    h, w = temp_map.shape

    # ---------------------------------------------------------------
    # 1. CREATE CANDIDATE MASK
    # ---------------------------------------------------------------

    candidate_mask = create_ohe_candidate_mask(
        color_img
    )

    if np.count_nonzero(candidate_mask) == 0:

        return {
            "wire_t_max": None,
            "wire_t_min": None,
            "delta_t": None,
            "alert": "No OHE structure detected",
            "wire_mask": candidate_mask
        }

    # ---------------------------------------------------------------
    # 2. CONNECTED COMPONENT SELECTION
    # ---------------------------------------------------------------

    wire_mask = select_ohe_components(
        candidate_mask
    )

    if np.count_nonzero(wire_mask) == 0:

        return {
            "wire_t_max": None,
            "wire_t_min": None,
            "delta_t": None,
            "alert": "No OHE structure detected",
            "wire_mask": wire_mask
        }

    # ---------------------------------------------------------------
    # 3. MASK REFINEMENT
    # ---------------------------------------------------------------

    wire_mask = refine_ohe_mask(
        wire_mask,
        color_img
    )

    # ---------------------------------------------------------------
    # 4. EXTRACT TEMPERATURES
    # ---------------------------------------------------------------

    wire_temps = temp_map[
        wire_mask == 1
    ]

    wire_temps = wire_temps[
        np.isfinite(wire_temps)
    ]

    if wire_temps.size < 20:

        return {
            "wire_t_max": None,
            "wire_t_min": None,
            "delta_t": None,
            "alert": "Insufficient valid OHE pixels",
            "wire_mask": wire_mask
        }

    # ---------------------------------------------------------------
    # 5. CLIP TO DETECTED SCALE
    # ---------------------------------------------------------------

    scale_low = min(
        float(t_max_scale),
        float(t_min_scale)
    )

    scale_high = max(
        float(t_max_scale),
        float(t_min_scale)
    )

    wire_temps = np.clip(
        wire_temps,
        scale_low,
        scale_high
    )

    # ---------------------------------------------------------------
    # 6. REMOVE EXTREME OUTLIERS
    #
    # Raw minimum can be caused by a few anti-aliased edge pixels.
    # Raw maximum can also contain isolated mapping noise.
    #
    # Therefore use robust percentiles.
    # ---------------------------------------------------------------

    wire_t_min = float(
        np.percentile(
            wire_temps,
            5
        )
    )

    wire_t_max = float(
        np.percentile(
            wire_temps,
            99
        )
    )

    # ---------------------------------------------------------------
    # 7. SAFETY CHECK
    # ---------------------------------------------------------------

    if wire_t_max < wire_t_min:

        return {
            "wire_t_max": None,
            "wire_t_min": None,
            "delta_t": None,
            "alert": "Invalid temperature range",
            "wire_mask": wire_mask
        }

    # ---------------------------------------------------------------
    # 8. DELTA T
    # ---------------------------------------------------------------

    delta_t = (
        wire_t_max
        -
        wire_t_min
    )

    # ---------------------------------------------------------------
    # 9. FAULT CLASSIFICATION
    # ---------------------------------------------------------------

    if delta_t > 20:

        alert = (
            "CRITICAL - Attend within 24 hrs"
        )

    elif delta_t > 10:

        alert = (
            "WARNING - Attend within 10 days"
        )

    elif delta_t > 5:

        alert = (
            "MONITOR - Attend within 1 month"
        )

    else:

        alert = "NORMAL"

    return {
        "wire_t_max": wire_t_max,
        "wire_t_min": wire_t_min,
        "delta_t": delta_t,
        "alert": alert,
        "wire_mask": wire_mask
    }


# ═══════════════════════════════════════════════════════════════════
# STATION LOOKUP
# ═══════════════════════════════════════════════════════════════════

def get_station_from_filename(
    image_filename,
    excel_path=None
):
    """
    Match timestamp in image filename to nearest OHE mast/station
    in the reference Google Sheet.
    """

    try:

        basename = os.path.splitext(
            os.path.basename(
                image_filename
            )
        )[0]

        parts = basename.split("-")

        if len(parts) < 2:
            return None

        time_str = parts[1]

        img_time = datetime.strptime(
            time_str,
            "%H%M%S"
        ).time()

        SHEET_ID = (
            "13W4XDKVK384EfZ5rxtccApLsMkca_Jz22qzz-uyrHf8"
        )

        url = (
            "https://docs.google.com/spreadsheets/d/"
            f"{SHEET_ID}/export?format=csv"
        )

        df = pd.read_csv(
            url
        )

        def find_col(
            df,
            keywords
        ):

            for col in df.columns:

                col_lower = str(
                    col
                ).lower()

                if any(
                    kw in col_lower
                    for kw in keywords
                ):

                    return col

            return None

        col_section = find_col(
            df,
            [
                "section",
                "station",
                "name"
            ]
        )

        col_ohe = find_col(
            df,
            [
                "ohe",
                "mast"
            ]
        )

        col_datetime = find_col(
            df,
            [
                "date",
                "time",
                "datetime"
            ]
        )

        if (
            not col_datetime
            or
            not col_section
        ):
            return None

        def parse_dt(val):

            s = (
                str(val)
                .strip()
                .replace(
                    " UTC",
                    ""
                )
            )

            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%d/%m/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M:%S",
                "%Y-%m-%d"
            ]

            for fmt in formats:

                try:

                    return datetime.strptime(
                        s,
                        fmt
                    )

                except Exception:
                    continue

            return None

        df["parsed_dt"] = df[
            col_datetime
        ].apply(
            parse_dt
        )

        df = df.dropna(
            subset=[
                "parsed_dt"
            ]
        )

        if df.empty:
            return None

        def to_secs(t):

            return (
                t.hour * 3600
                +
                t.minute * 60
                +
                t.second
            )

        img_secs = to_secs(
            img_time
        )

        df["diff_secs"] = df[
            "parsed_dt"
        ].apply(
            lambda dt:
                abs(
                    to_secs(
                        dt.time()
                    )
                    -
                    img_secs
                )
        )

        nearest = df.loc[
            df["diff_secs"].idxmin()
        ]

        if nearest["diff_secs"] <= 300:

            ohe_raw = (
                nearest[col_ohe]
                if col_ohe
                else "N/A"
            )

            ohe_str = (
                str(ohe_raw)
                .strip()
                .replace(
                    ".0",
                    ""
                )
            )

            return {
                "section":
                    str(
                        nearest[
                            col_section
                        ]
                    ).strip(),

                "ohe_mast":
                    ohe_str,

                "matched_time":
                    nearest[
                        "parsed_dt"
                    ].strftime(
                        "%H:%M:%S"
                    ),

                "diff_seconds":
                    int(
                        nearest[
                            "diff_secs"
                        ]
                    )
            }

    except Exception as e:

        print(
            f"[station lookup error] {e}"
        )

    return None


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def process_image(image_path):
    """
    Complete thermal image analysis pipeline.

    No manual ROI selection is required.
    """

    # ---------------------------------------------------------------
    # 1. LOAD IMAGE
    # ---------------------------------------------------------------

    color_img = cv2.imread(
        image_path
    )

    if color_img is None:

        raise ValueError(
            f"Cannot load image: {image_path}"
        )

    h, w = color_img.shape[:2]

    # ---------------------------------------------------------------
    # 2. EXTRACT THERMAL SCALE
    # ---------------------------------------------------------------
    #
    # The thermal colour scale is normally on the right.
    # Keep a small margin around it for OCR.
    # ---------------------------------------------------------------

    scale_x1 = int(
        w * 0.90
    )

    scale_x2 = int(
        w * 0.98
    )

    scale = color_img[
        :,
        scale_x1:scale_x2
    ]

    if scale.size == 0:

        raise ValueError(
            "Could not extract thermal scale."
        )

    sh, sw = scale.shape[:2]

    # ---------------------------------------------------------------
    # 3. OCR TOP / BOTTOM TEMPERATURE
    # ---------------------------------------------------------------

    top_crop = scale[
        int(sh * 0.20):
        int(sh * 0.28),
        :
    ]

    bottom_crop = scale[
        int(sh * 0.78):
        int(sh * 0.86),
        :
    ]

    t_max_abs = parse_scale_val(
        top_crop
    )

    t_min_abs = parse_scale_val(
        bottom_crop
    )

    # ---------------------------------------------------------------
    # 4. PRESERVE NEGATIVE SCALE
    # ---------------------------------------------------------------
    #
    # Do NOT blindly swap values.
    #
    # Example:
    #
    #     top  = 31
    #     bottom = -28
    #
    # This is a valid scale and must remain:
    #
    #     t_max = 31
    #     t_min = -28
    # ---------------------------------------------------------------

    t_max = t_max_abs
    t_min = t_min_abs

    # ---------------------------------------------------------------
    # 5. OCR FALLBACK
    #
    # These are only fallbacks if OCR completely fails.
    # ---------------------------------------------------------------

    if t_max is None:

        t_max = 35.0

    if t_min is None:

        t_min = -26.0

    # ---------------------------------------------------------------
    # 6. VALIDATE SCALE
    # ---------------------------------------------------------------

    if (
        not np.isfinite(t_max)
        or
        not np.isfinite(t_min)
    ):

        raise ValueError(
            "Invalid thermal scale detected."
        )

    if t_max == t_min:

        raise ValueError(
            "Thermal scale maximum and minimum are identical."
        )

    # Do NOT swap t_max/t_min blindly.
    #
    # The top of the thermal scale should represent maximum
    # temperature and the bottom should represent minimum.
    #
    # If OCR appears inconsistent, report it rather than silently
    # changing the values.

    if t_max < t_min:

        raise ValueError(
            "Detected thermal scale is inconsistent: "
            f"maximum={t_max}, minimum={t_min}. "
            "Please verify the scale OCR."
        )

    # ---------------------------------------------------------------
    # 7. BUILD TEMPERATURE MAP
    # ---------------------------------------------------------------

    temp_map = map_pixels_to_temperature(
        color_img,
        scale,
        t_max,
        t_min
    )

    # ---------------------------------------------------------------
    # 8. AUTOMATIC OHE ANALYSIS
    # ---------------------------------------------------------------

    result = segment_wire_and_compute_delta_t(
        temp_map,
        t_max,
        t_min,
        color_img
    )

    # ---------------------------------------------------------------
    # 9. RETURN STRUCTURED RESULT
    # ---------------------------------------------------------------

    return {
        "scale_t_max":
            round(
                float(t_max),
                1
            ),

        "scale_t_min":
            round(
                float(t_min),
                1
            ),

        "max_temp":
            (
                round(
                    float(
                        result["wire_t_max"]
                    ),
                    1
                )
                if result["wire_t_max"] is not None
                else None
            ),

        "min_temp":
            (
                round(
                    float(
                        result["wire_t_min"]
                    ),
                    1
                )
                if result["wire_t_min"] is not None
                else None
            ),

        "delta":
            (
                round(
                    float(
                        result["delta_t"]
                    ),
                    1
                )
                if result["delta_t"] is not None
                else None
            ),

        "status":
            result["alert"],

        "temp_map":
            temp_map,

        "wire_mask":
            result["wire_mask"]
    }
