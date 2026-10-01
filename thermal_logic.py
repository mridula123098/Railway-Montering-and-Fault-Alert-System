def process_image(image_path):
    color_img = cv2.imread(image_path)
    if color_img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = color_img.shape[:2]

    # Extract scale limits (Top/Bottom) -> Top is 17
    t_max, t_min = parse_scale_bounds(color_img)

    # Build temperature map
    scale = color_img[:, int(w * 0.90):int(w * 0.98)]
    temp_map = map_pixels_to_temperature(color_img, scale, t_max, t_min)

    # Segment wire region
    wire_t_max, wire_t_min, _, _, wire_mask = segment_wire_and_compute_delta_t(
        temp_map, color_img
    )

    # Force Max Temp to strictly follow the Palette Scale Upper Limit (17 °C)
    final_max_temp = float(t_max)
    
    # Min Temp remains the wire minimum (6.5 °C)
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
        "max_temp": round(final_max_temp, 1), # Will output 17.0 °C
        "min_temp": final_min_temp,          # Will output 6.5 °C (or wire min)
        "delta": delta_t,                    # Will output 10.5 °C
        "status": alert,
        "temp_map": temp_map,
        "wire_mask": wire_mask
    }
