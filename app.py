import streamlit as st
import tempfile

from thermal_logic import process_image


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Thermal Fault Detection",
    layout="wide"
)

st.title("Railway Thermal Fault Detection System")

st.write(
    "Upload a thermal image for analysis."
)


# =========================================
# FILE UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "Upload Thermal Image",
    type=["jpg", "jpeg", "png"]
)


# =========================================
# PROCESS IMAGE
# =========================================

if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:

        tmp.write(uploaded_file.read())

        image_path = tmp.name

    # Display uploaded image
    st.image(
        image_path,
        caption="Uploaded Thermal Image",
        use_container_width=True
    )

    # Processing animation
    with st.spinner("Analyzing thermal image..."):

        result = process_image(image_path)

    st.success("Analysis Complete")


    # =====================================
    # DISPLAY RESULTS
    # =====================================

    st.subheader("Temperature Scale")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Scale Maximum Temperature",
            f"{result['scale_t_max']:.1f} °C"
        )

    with col2:

        st.metric(
            "Scale Minimum Temperature",
            f"{result['scale_t_min']:.1f} °C"
        )


    st.subheader("Wire Temperature Analysis")

    col3, col4 = st.columns(2)

    with col3:

        st.metric(
            "Wire Maximum Temperature",
            f"{result['max_temp']:.1f} °C"
        )

    with col4:

        st.metric(
            "Wire Minimum Temperature",
            f"{result['min_temp']:.1f} °C"
        )


    st.metric(
        "Delta T",
        f"{result['delta']:.1f} °C"
    )


    # =====================================
    # STATUS ALERT
    # =====================================

    st.subheader("Fault Status")

    if "CRITICAL" in result["status"]:

        st.error(result["status"])

    elif "WARNING" in result["status"]:

        st.warning(result["status"])

    elif "MONITOR" in result["status"]:

        st.info(result["status"])

    else:

        st.success(result["status"])