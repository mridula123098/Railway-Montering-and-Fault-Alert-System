"""
AI-based OHE Wire Fault Detection System

Dashboard Development:
Mridula K.

Modules:
- Streamlit UI
- Station Matching Engine
- Thermal Analysis Integration
- SQLite Reporting System

Date: June 2026
"""

import streamlit as st
import tempfile
import os
import base64
import pandas as pd
from datetime import datetime
from thermal_logic import process_image, get_station_from_filename
from database import save_report
# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="KRC Thermal Fault Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════
# LOGO
# ═══════════════════════════════════════════════════════════════════

def get_logo_base64():
    logo_path = os.path.join(os.path.dirname(__file__), "konkan_logo.png")
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

logo_b64  = get_logo_base64()
logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" '
    f'style="width:80px;height:80px;object-fit:contain;">'
    if logo_b64
    else '<div class="krc-logo-placeholder">logo<br>here</div>'
)

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer     {visibility:hidden;}
header     {visibility:hidden;}

.stApp {
    background-color: white;
    min-height: 100vh;
}

/* Remove default streamlit padding */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Header ── */
.krc-header {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 14px 24px;
    gap: 18px;
    border-bottom: 2px solid #3a007a;
    background: white;
    width: 100%;
}
.krc-logo-placeholder {
    border: 2px dashed #888;
    width: 64px; height: 52px;
    display: flex; align-items: center; justify-content: center;
    font-size: 9px; color: #888; text-align: center;
    line-height: 1.4; flex-shrink: 0;
}
.krc-org {
    font-size: 26px;
    font-weight: 900;
    color: #00008b;
    font-family: 'Arial Black', Arial, sans-serif;
}

/* ── Subtitle ── */
.krc-subtitle {
    text-align: center;
    padding: 10px;
    font-weight: 700;
    font-size: 14px;
    color: #111;
    border-bottom: 1px solid #ddd;
    background: white;
    letter-spacing: 0.5px;
    font-family: 'Courier New', monospace;
}

/* ── Content area — centered column ── */
.krc-content {
    max-width: 800px;
    margin: 24px auto;
    padding: 0 24px;
}

/* ── Info / station badges ── */
.info-badge {
    background: #f5f8ff;
    border: 1px solid #c0caee;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 13px;
    font-family: Arial, sans-serif;
    color: #222;
    margin: 8px 0;
}
.info-badge b { color: #00008b; }
.station-badge {
    background: #f5f8ff;
    border: 1px solid #c0caee;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 13px;
    font-family: Arial, sans-serif;
    color: #222;
    margin: 8px 0;
}
.station-badge span { color: #00008b; font-weight: 700; }

/* ── File uploader — full width, centered content ── */
div[data-testid="stFileUploader"] {
    width: 100% !important;
}
div[data-testid="stFileUploader"] section {
    background: #00008b !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 14px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    min-height: 60px !important;
}
div[data-testid="stFileUploader"] section > div {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 10px !important;
}
div[data-testid="stFileUploader"] section * {
    color: white !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
div[data-testid="stFileUploader"] section svg {
    fill: white !important;
    stroke: white !important;
}

/* ── Analyse button — centered ── */
div[data-testid="stButton"] {
    display: flex !important;
    justify-content: center !important;
    margin-top: 16px !important;
}
div[data-testid="stButton"] > button {
    background-color: #00008b !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 10px 48px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    font-family: Arial, sans-serif !important;
    letter-spacing: 0.5px !important;
}
div[data-testid="stButton"] > button:hover {
    background-color: #0000cc !important;
}

/* ── Report divider ── */
.krc-divider {
    border: none;
    border-top: 2px solid #3a007a;
    margin: 24px 0 20px;
}

/* ── Report title ── */
.report-title {
    text-align: center;
    font-size: 16px;
    font-weight: 900;
    color: #111;
    margin-bottom: 20px;
    font-family: Arial, sans-serif;
    text-decoration: underline;
}

/* ── Metric rows ── */
.metric-row {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
    gap: 16px;
    font-family: Arial, sans-serif;
}
.metric-key {
    font-size: 13px;
    font-weight: 700;
    color: #111;
    min-width: 220px;
    text-align: right;
}
.metric-val {
    padding: 6px 24px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 700;
    color: white;
    min-width: 180px;
    text-align: center;
}
.val-blue   { background: #00008b; }
.val-green  { background: #1a7a1a; }
.val-yellow { background: #b8860b; }
.val-red    { background: #cc0000; }

/* ── Attend line ── */
.attend-line {
    text-align: center;
    font-size: 14px;
    font-weight: 700;
    color: #111;
    margin-top: 20px;
    padding-top: 14px;
    border-top: 1px solid #ddd;
    font-family: Arial, sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HEADER — full width
# ═══════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="krc-header">
    {logo_html}
    <div class="krc-org">Konkan Railway Corporation Limited</div>
</div>
<div class="krc-subtitle">AI-based OHE Wire Fault Detection System</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CENTERED CONTENT — use a narrow center column for content
# ═══════════════════════════════════════════════════════════════════

_, center, _ = st.columns([1, 3, 1])

with center:

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        # Image preview
        st.image(uploaded_file,
                 caption="Uploaded Thermal Image",
                 use_container_width=True)

        # Extract date & time from filename
        basename       = os.path.splitext(uploaded_file.name)[0]
        parts          = basename.split("-")
        extracted_date = "Unknown"
        extracted_time = "Unknown"
        img_dt_obj     = None

        if len(parts) >= 2:
            try:
                img_dt_obj     = datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M%S")
                extracted_date = img_dt_obj.strftime("%d/%m/%Y")
                extracted_time = img_dt_obj.strftime("%H:%M:%S")
            except Exception:
                pass

        st.markdown(f"""
        <div class="info-badge">
            🕐 &nbsp; Date: <b>{extracted_date}</b>
            &nbsp;|&nbsp;
            Time: <b>{extracted_time}</b>
        </div>
        """, unsafe_allow_html=True)

        # Station lookup
        excel_path = os.path.join(os.path.dirname(__file__), "testing_sheet.xlsx")
        station    = get_station_from_filename(uploaded_file.name, excel_path)

        if station:

            if station["diff_seconds"] <= 10:

                st.markdown(f"""
                <div class="station-badge">
                    📍 Section:
                    <b>{station['section']}</b>

                    &nbsp; | &nbsp;

                    OHE Mast:
                    <b>{station['ohe_mast']}</b>

                    &nbsp; | &nbsp;

                    Matched:
                    <b>{station['matched_time']}</b>

                    &nbsp;

                    (±{station['diff_seconds']}s)
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="station-badge">
                    ⚠️ No station match found.
                </div>
                """, unsafe_allow_html=True)

        else:
            try:
                df = pd.read_excel(excel_path, header=0)

                def find_col(df, keywords):
                    for col in df.columns:
                        if any(kw in str(col).lower() for kw in keywords):
                            return col
                    return None

                col_dt = find_col(df, ["date", "time"])

                def parse_dt(val):
                    s = str(val).strip().replace(" UTC", "")
                    for fmt in ["%Y-%m-%d %H:%M:%S",
                                "%d/%m/%Y %H:%M:%S",
                                "%m/%d/%Y %H:%M:%S"]:
                        try:
                            return datetime.strptime(s, fmt)
                        except Exception:
                            continue
                    return None

                if col_dt:
                    df["parsed_dt"] = df[col_dt].apply(parse_dt)
                    df = df.dropna(subset=["parsed_dt"])

                    def to_secs(t):
                        return t.hour * 3600 + t.minute * 60 + t.second

                    if img_dt_obj and not df.empty:
                        img_secs        = to_secs(img_dt_obj.time())
                        df["diff_secs"] = df["parsed_dt"].apply(
                            lambda dt: abs(to_secs(dt.time()) - img_secs)
                        )
                        nearest      = df.loc[df["diff_secs"].idxmin()]
                        nearest_secs = int(nearest["diff_secs"])
                        col_sec      = find_col(df, ["section", "station"])
                        col_ohe      =find_col(df,["ohe mast","ohe_mast"])
                        sec_name     = str(nearest[col_sec]) if col_sec else "Unknown"
                        # ohe_mast_name= str(nearest[col_ohe]) if col_ohe else "Unknown"
                        ohe_val = nearest[col_ohe]

                        from datetime import datetime

                        if isinstance(ohe_val, (pd.Timestamp, datetime)):
                            ohe_mast_name = f"{ohe_val.day}/{ohe_val.month}"
                        else:
                            ohe_mast_name = str(ohe_val)
                       
                        st.markdown(f"""
                        <div class="station-badge" style="color:#888;">
                             Nearest:
                            <b style="color:#333">{sec_name} {ohe_mast_name}</b>
                            at <b style="color:#333">
                            {nearest['parsed_dt'].strftime('%H:%M:%S')}</b>
                            (diff = {nearest_secs//60}m {nearest_secs%60}s)
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="station-badge" style="color:#888;">
                            ⚠️ Could not parse filename for station lookup.
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="station-badge" style="color:#c00;">
                    ❌ Station lookup error: {e}
                </div>
                """, unsafe_allow_html=True)


    # Center the button using columns inside center column
    b1, b2, b3 = st.columns([2, 1, 2])
    with b2:
        analyse_clicked = st.button(
            "Analyse",
            disabled=(uploaded_file is None),
            use_container_width=True
        )
# ═══════════════════════════════════════════════════════════════════
# ANALYSIS RESULTS
# ═══════════════════════════════════════════════════════════════════

if analyse_clicked and uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    with center:
        with st.spinner("Analysing thermal image..."):
            result = process_image(image_path)
            
    status = result["status"]   # or result["status"] depending on thermal_logic.py
    from datetime import datetime
    
    if "CRITICAL" in status:
        val_class = "val-red"
        attend_msg = "To be attended in 1 day"
    
    elif "WARNING" in status:
        val_class = "val-yellow"
        attend_msg = "To be attended in 10 days"
    
    elif "MONITOR" in status:
        val_class = "val-yellow"
        attend_msg = "To be attended in 30 days"
    
    else:
        val_class = "val-green"
        attend_msg = "Normal — No fault detected"
    st.write("Station object:", station)
    save_report({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": uploaded_file.name,
        "capture_date": extracted_date,
        "capture_time": extracted_time,
        "section": station["section"] if station else "",
        "ohe_mast": station["ohe_mast"] if station else "",
        "scale_max": result["scale_t_max"],
        "scale_min": result["scale_t_min"],
        "wire_max": result["max_temp"],
        "wire_min": result["min_temp"],
        "delta_t": result["delta"],
        "status": status,
        "attend_in": attend_msg
    })
# # =====================================
# # SAVE TO DATABASE
# # =====================================

# try:

#     if station:

#         save_report(
#             image_name=uploaded_file.name,

#             section=station["section"],
#             ohe_mast=station["ohe_mast"],

#             scale_max=result["scale_t_max"],
#             scale_min=result["scale_t_min"],

#             wire_max=result["max_temp"],
#             wire_min=result["min_temp"],

#             delta_t=result["delta"],
#             status=result["status"]
#         )

# except Exception as e:
#     print("Database Save Error:", e)


#     os.unlink(image_path)

#     if result["max_temp"] is None:
#         with center:
#             st.error("Could not detect wire region. Please try another image.")
#     else:
#         status = result["status"]
#         delta  = result["delta"]

#         if "CRITICAL" in status:
#             val_class  = "val-red"
#             attend_msg = "To be attended in <b>1</b> day"
#         elif "WARNING" in status:
#             val_class  = "val-yellow"
#             attend_msg = "To be attended in <b>10</b> days"
#         elif "MONITOR" in status:
#             val_class  = "val-yellow"
#             attend_msg = "To be attended in <b>30</b> days"
#         else:
#             val_class  = "val-green"
#             attend_msg = "&#10003; &nbsp; Normal — No fault detected"

#         st.markdown(f"""
#         <div style="max-width:800px;margin:0 auto;padding:0 24px;">
#           <hr class="krc-divider">
#           <div class="report-title">Analysis Report</div>

#           <div class="metric-row">
#             <span class="metric-key">Scale Max Temperature :</span>
#             <span class="metric-val val-blue">{result['scale_t_max']:.1f} °C</span>
#           </div>
#           <div class="metric-row">
#             <span class="metric-key">Scale Min Temperature :</span>
#             <span class="metric-val val-blue">{result['scale_t_min']:.1f} °C</span>
#           </div>
#           <div class="metric-row">
#             <span class="metric-key">Max Temperature :</span>
#             <span class="metric-val {val_class}">{result['max_temp']:.1f} °C</span>
#           </div>
#           <div class="metric-row">
#             <span class="metric-key">Min Temperature :</span>
#             <span class="metric-val {val_class}">{result['min_temp']:.1f} °C</span>
#           </div>
#           <div class="metric-row">
#             <span class="metric-key">Temperature Diff :</span>
#             <span class="metric-val {val_class}">{delta:.1f} °C</span>
#           </div>

#           <div class="attend-line">{attend_msg}</div>
#         </div>
#         """, unsafe_allow_html=True)
    # =====================================
    # SAVE TO DATABASE
    # =====================================
    # try:
       
    #     if station:

    #         save_report(
    #             image_name=uploaded_file.name,

    #             section=station["section"],
    #             ohe_mast=station["ohe_mast"],

    #             scale_max=result["scale_t_max"],
    #             scale_min=result["scale_t_min"],

    #             wire_max=result["max_temp"],
    #             wire_min=result["min_temp"],

    #             delta_t=result["delta"],
    #             status=result["status"]
    #         )

    # except Exception as e:
    #     print("Database Save Error:", e)

    # delete temp image
    os.unlink(image_path)

    if result["max_temp"] is None:

        with center:
            st.error(
                "Could not detect wire region. Please try another image."
            )

    else:

        status = result["status"]
        delta = result["delta"]

        if "CRITICAL" in status:
            val_class = "val-red"
            attend_msg = "To be attended in <b>1</b> day"

        elif "WARNING" in status:
            val_class = "val-yellow"
            attend_msg = "To be attended in <b>10</b> days"

        elif "MONITOR" in status:
            val_class = "val-yellow"
            attend_msg = "To be attended in <b>30</b> days"

        else:
            val_class = "val-green"
            attend_msg = "&#10003; &nbsp; Normal — No fault detected"

        st.markdown(f"""
        <div style="max-width:800px;margin:0 auto;padding:0 24px;">
          <hr class="krc-divider">
          <div class="report-title">Analysis Report</div>

          <div class="metric-row">
            <span class="metric-key">Scale Max Temperature :</span>
            <span class="metric-val val-blue">{result['scale_t_max']:.1f} °C</span>
          </div>

          <div class="metric-row">
            <span class="metric-key">Scale Min Temperature :</span>
            <span class="metric-val val-blue">{result['scale_t_min']:.1f} °C</span>
          </div>

          <div class="metric-row">
            <span class="metric-key">Max Temperature :</span>
            <span class="metric-val {val_class}">{result['max_temp']:.1f} °C</span>
          </div>

          <div class="metric-row">
            <span class="metric-key">Min Temperature :</span>
            <span class="metric-val {val_class}">{result['min_temp']:.1f} °C</span>
          </div>

          <div class="metric-row">
            <span class="metric-key">Temperature Diff :</span>
            <span class="metric-val {val_class}">{delta:.1f} °C</span>
          </div>

          <div class="attend-line">{attend_msg}</div>
        </div>
        """, unsafe_allow_html=True)
