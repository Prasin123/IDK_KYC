import json
import re
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
import streamlit as st

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "PDF OCR Information Verifier"
ROI_FILE = Path("roi_config.json")

# If Tesseract is not in PATH on Windows, uncomment and edit:
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 PDF OCR Information Verifier")
st.caption(
    "Upload two PDFs, define named ROIs, OCR the same regions, and compare the results."
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "rois": {},
    "pdf_a_bytes": None,
    "pdf_b_bytes": None,
    "last_draw_signature": None,
    "last_results": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# ROI FILE FUNCTIONS
# ============================================================

def load_roi_config():
    if not ROI_FILE.exists():
        return {}

    try:
        with ROI_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def save_roi_config():
    try:
        with ROI_FILE.open("w", encoding="utf-8") as f:
            json.dump(
                st.session_state.rois,
                f,
                indent=4,
            )
        return True
    except OSError as exc:
        st.error(f"Could not save ROI configuration: {exc}")
        return False


if not st.session_state.rois:
    st.session_state.rois = load_roi_config()


# ============================================================
# PDF FUNCTIONS
# ============================================================

def open_pdf(pdf_bytes):
    """Open a PDF from bytes."""
    if not pdf_bytes:
        return None

    return fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )


def render_page(pdf, page_number, zoom=2.0):
    """Render one PDF page as a PIL image."""
    page = pdf.load_page(page_number)

    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    return Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples,
    )


# ============================================================
# IMAGE / OCR
# ============================================================

def preprocess_image(image):
    """PIL-only preprocessing; no OpenCV required."""
    image = ImageOps.grayscale(image)

    width, height = image.size

    image = image.resize(
        (max(1, width * 2), max(1, height * 2)),
        Image.Resampling.LANCZOS,
    )

    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    image = ImageOps.autocontrast(image)

    return image


def normalize_text(text):
    """Basic normalization for OCR comparison."""
    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_ocr_text(text):
    """Clean common OCR artifacts."""
    text = normalize_text(text)
    text = re.sub(r"[|_~]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ocr_with_confidence(image, language="eng"):
    """Run Tesseract and return text + average confidence."""
    processed = preprocess_image(image)

    config = "--oem 3 --psm 6"

    try:
        data = pytesseract.image_to_data(
            processed,
            lang=language,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError as exc:
        raise RuntimeError(
            f"Tesseract OCR failed. Check that Tesseract and the "
            f"'{language}' language data are installed. Details: {exc}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Tesseract was not found. Install Tesseract or set "
            "pytesseract.pytesseract.tesseract_cmd near the top of app.py."
        ) from exc

    words = []
    confidences = []

    for i, word in enumerate(data.get("text", [])):
        word = word.strip()

        if not word:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError, IndexError):
            confidence = -1

        words.append(word)

        if confidence >= 0:
            confidences.append(confidence)

    text = clean_ocr_text(" ".join(words))

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )

    return text, average_confidence


# ============================================================
# ROI FUNCTIONS
# ============================================================

def get_page_rois(page_number):
    """Get/create ROI dictionary for a page."""
    page_key = str(page_number)

    if page_key not in st.session_state.rois:
        st.session_state.rois[page_key] = {}

    return st.session_state.rois[page_key]


def crop_roi(image, roi):
    """Crop a normalized ROI from the original image."""
    width, height = image.size

    x = int(round(roi["x"] * width))
    y = int(round(roi["y"] * height))

    w = int(round(roi["width"] * width))
    h = int(round(roi["height"] * height))

    x = max(0, min(x, width))
    y = max(0, min(y, height))
    x2 = max(x, min(x + w, width))
    y2 = max(y, min(y + h, height))

    return image.crop((x, y, x2, y2))


def normalized_roi_from_points(x1, y1, x2, y2, display_width, display_height):
    """Convert display coordinates into normalized 0..1 ROI coordinates."""
    left = max(0, min(x1, x2))
    top = max(0, min(y1, y2))
    right = min(display_width, max(x1, x2))
    bottom = min(display_height, max(y1, y2))

    if right <= left or bottom <= top:
        return None

    return {
        "x": left / display_width,
        "y": top / display_height,
        "width": (right - left) / display_width,
        "height": (bottom - top) / display_height,
    }


def draw_roi_preview(image, rois):
    """Draw all saved ROIs on a preview image."""
    preview = image.copy()
    draw = ImageDraw.Draw(preview)

    width, height = preview.size

    for index, (name, roi) in enumerate(rois.items(), start=1):
        x1 = int(roi["x"] * width)
        y1 = int(roi["y"] * height)
        x2 = int((roi["x"] + roi["width"]) * width)
        y2 = int((roi["y"] + roi["height"]) * height)

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=max(2, width // 500),
        )

        label = f"{index}. {name}"

        # Small label background.
        try:
            bbox = draw.textbbox((0, 0), label)
            label_w = bbox[2] - bbox[0]
            label_h = bbox[3] - bbox[1]
        except AttributeError:
            label_w, label_h = 100, 20

        label_x = x1
        label_y = max(0, y1 - label_h - 4)

        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + label_w + 8,
                label_y + label_h + 4,
            ),
            fill="white",
            outline="red",
        )

        draw.text(
            (label_x + 4, label_y + 2),
            label,
            fill="red",
        )

    return preview


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

language = st.sidebar.selectbox(
    "OCR Language",
    ["eng", "eng+nep"],
    index=0,
)

st.sidebar.caption(
    "Use eng for English or eng+nep for English + Nepali."
)

st.sidebar.divider()

if st.sidebar.button("🗑️ Clear all saved ROIs", use_container_width=True):
    st.session_state.rois = {}
    st.session_state.last_draw_signature = None
    save_roi_config()
    st.rerun()


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("1. Upload the two PDFs")

col1, col2 = st.columns(2)

with col1:
    uploaded_a = st.file_uploader(
        "PDF A",
        type=["pdf"],
        key="pdf_a_upload",
    )

with col2:
    uploaded_b = st.file_uploader(
        "PDF B",
        type=["pdf"],
        key="pdf_b_upload",
    )


# Keep the raw bytes in session state so the PDFs survive Streamlit reruns.
if uploaded_a is not None:
    st.session_state.pdf_a_bytes = uploaded_a.getvalue()

if uploaded_b is not None:
    st.session_state.pdf_b_bytes = uploaded_b.getvalue()


if not st.session_state.pdf_a_bytes or not st.session_state.pdf_b_bytes:
    st.info("Upload both PDFs to start.")
    st.stop()


pdf_a = open_pdf(st.session_state.pdf_a_bytes)
pdf_b = open_pdf(st.session_state.pdf_b_bytes)

if pdf_a is None or pdf_b is None:
    st.error("Could not open one of the PDFs.")
    st.stop()


# ============================================================
# PAGE SELECTION
# ============================================================

st.subheader("2. Choose the page")

max_pages = min(len(pdf_a), len(pdf_b))

page_number = st.number_input(
    "Page number",
    min_value=1,
    max_value=max_pages,
    value=1,
    step=1,
)

page_index = page_number - 1

image_a = render_page(pdf_a, page_index, zoom=2.0)
image_b = render_page(pdf_b, page_index, zoom=2.0)

page_rois = get_page_rois(page_number)


# ============================================================
# ROI EDITOR
# ============================================================

st.subheader("3. ROI Editor")

left, right = st.columns([1, 3])

with left:
    st.markdown("### ROI Manager")

    roi_names = list(page_rois.keys())

    if roi_names:
        selected_roi = st.selectbox(
            "Select ROI",
            roi_names,
            key=f"selected_roi_{page_number}",
        )
    else:
        selected_roi = None
        st.info("Create your first ROI.")

    st.divider()

    new_roi_name = st.text_input(
        "New ROI name",
        placeholder="e.g. full_name",
        key=f"new_roi_name_{page_number}",
    )

    if st.button(
        "➕ Add ROI",
        use_container_width=True,
        key=f"add_roi_{page_number}",
    ):
        clean_name = new_roi_name.strip()

        if not clean_name:
            st.error("Enter an ROI name.")
        elif clean_name in page_rois:
            st.error("That ROI already exists on this page.")
        else:
            page_rois[clean_name] = {
                "x": 0.10,
                "y": 0.10,
                "width": 0.20,
                "height": 0.08,
            }
            save_roi_config()
            st.session_state.last_draw_signature = None
            st.rerun()

    if selected_roi:
        st.divider()

        if st.button(
            "🗑️ Delete selected ROI",
            use_container_width=True,
            key=f"delete_roi_{page_number}_{selected_roi}",
        ):
            del page_rois[selected_roi]
            save_roi_config()
            st.session_state.last_draw_signature = None
            st.rerun()


with right:
    st.markdown(
        "Select an ROI name, then **click and drag** over the field in PDF A."
    )

    # The component uses this exact option for drag selection.
    display_width = 1000
    display_height = max(
        1,
        int(round(image_a.height * display_width / image_a.width)),
    )

    coords = streamlit_image_coordinates(
        image_a,
        width=display_width,
        click_and_drag=True,
        cursor="crosshair",
        key=f"roi_canvas_{page_number}",
    )

    if coords and selected_roi:
        required_keys = {"x1", "y1", "x2", "y2"}

        if required_keys.issubset(coords.keys()):
            x1 = float(coords["x1"])
            y1 = float(coords["y1"])
            x2 = float(coords["x2"])
            y2 = float(coords["y2"])

            signature = (
                selected_roi,
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2),
            )

            # Avoid repeatedly saving the same drag event on reruns.
            if signature != st.session_state.last_draw_signature:
                new_roi = normalized_roi_from_points(
                    x1,
                    y1,
                    x2,
                    y2,
                    display_width,
                    display_height,
                )

                if new_roi and new_roi["width"] >= 0.005 and new_roi["height"] >= 0.005:
                    page_rois[selected_roi] = new_roi
                    save_roi_config()
                    st.session_state.last_draw_signature = signature
                    st.success(f"Saved ROI: {selected_roi}")

        else:
            st.warning(
                "The image component did not return x1/y1/x2/y2. "
                "Update streamlit-image-coordinates if needed."
            )

    elif coords and not selected_roi:
        st.warning("Create and select an ROI before drawing.")


# ============================================================
# FINE ROI ADJUSTMENT
# ============================================================

if selected_roi:
    st.divider()
    st.subheader(f"Fine adjustment — {selected_roi}")

    roi = page_rois[selected_roi]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        x = st.number_input(
            "X",
            min_value=0.0,
            max_value=1.0,
            value=float(roi["x"]),
            step=0.001,
            format="%.4f",
            key=f"x_{page_number}_{selected_roi}",
        )

    with c2:
        y = st.number_input(
            "Y",
            min_value=0.0,
            max_value=1.0,
            value=float(roi["y"]),
            step=0.001,
            format="%.4f",
            key=f"y_{page_number}_{selected_roi}",
        )

    with c3:
        width = st.number_input(
            "Width",
            min_value=0.001,
            max_value=1.0,
            value=float(roi["width"]),
            step=0.001,
            format="%.4f",
            key=f"width_{page_number}_{selected_roi}",
        )

    with c4:
        height = st.number_input(
            "Height",
            min_value=0.001,
            max_value=1.0,
            value=float(roi["height"]),
            step=0.001,
            format="%.4f",
            key=f"height_{page_number}_{selected_roi}",
        )

    if st.button(
        "💾 Save adjusted ROI",
        use_container_width=True,
        key=f"save_adjusted_{page_number}_{selected_roi}",
    ):
        page_rois[selected_roi] = {
            "x": x,
            "y": y,
            "width": min(width, 1.0 - x),
            "height": min(height, 1.0 - y),
        }

        save_roi_config()
        st.session_state.last_draw_signature = None
        st.success("ROI updated.")
        st.rerun()


# ============================================================
# ROI PREVIEW
# ============================================================

st.divider()
st.subheader("4. Saved ROI Preview")

if page_rois:
    preview = draw_roi_preview(image_a, page_rois)
    st.image(
        preview,
        caption="PDF A with saved ROIs",
        use_container_width=True,
    )

    roi_table = []

    for name, roi in page_rois.items():
        roi_table.append(
            {
                "ROI": name,
                "X": round(roi["x"], 4),
                "Y": round(roi["y"], 4),
                "Width": round(roi["width"], 4),
                "Height": round(roi["height"], 4),
            }
        )

    st.dataframe(
        roi_table,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.warning("Create at least one ROI before running OCR.")


# ============================================================
# OCR COMPARISON
# ============================================================

st.divider()
st.subheader("5. Compare the PDFs")

if st.button(
    "🔎 OCR + Compare",
    type="primary",
    use_container_width=True,
):
    if not page_rois:
        st.error("Create at least one ROI first.")
        st.stop()

    results = []

    progress = st.progress(0.0)
    status = st.empty()

    total = len(page_rois)

    for index, (name, roi) in enumerate(page_rois.items(), start=1):
        status.write(f"OCR: {name}")

        crop_a = crop_roi(image_a, roi)
        crop_b = crop_roi(image_b, roi)

        try:
            text_a, confidence_a = ocr_with_confidence(
                crop_a,
                language,
            )

            text_b, confidence_b = ocr_with_confidence(
                crop_b,
                language,
            )

        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        normalized_a = normalize_text(text_a)
        normalized_b = normalize_text(text_b)

        match = bool(
            normalized_a
            and normalized_b
            and normalized_a == normalized_b
        )

        results.append(
            {
                "ROI": name,
                "PDF A": text_a,
                "PDF B": text_b,
                "Confidence A": round(confidence_a, 1),
                "Confidence B": round(confidence_b, 1),
                "Match": match,
            }
        )

        progress.progress(index / total)

    status.empty()
    st.session_state.last_results = results


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.last_results

if results:
    st.subheader("Results")

    matches = sum(1 for result in results if result["Match"])
    mismatches = len(results) - matches

    c1, c2, c3 = st.columns(3)

    c1.metric("Total ROIs", len(results))
    c2.metric("Matching", matches)
    c3.metric("Mismatching", mismatches)

    if mismatches == 0:
        st.success("✅ All non-empty OCR regions match.")
    else:
        st.error(f"❌ {mismatches} ROI(s) do not match.")

    st.divider()

    for result in results:
        name = result["ROI"]

        icon = "✅" if result["Match"] else "❌"

        with st.expander(
            f"{icon} {name}",
            expanded=not result["Match"],
        ):
            a, b = st.columns(2)

            with a:
                st.markdown("### PDF A")
                st.write(result["PDF A"] or "[No OCR text]")
                st.caption(
                    f"OCR confidence: {result['Confidence A']:.1f}%"
                )

            with b:
                st.markdown("### PDF B")
                st.write(result["PDF B"] or "[No OCR text]")
                st.caption(
                    f"OCR confidence: {result['Confidence B']:.1f}%"
                )


# ============================================================
# INDIVIDUAL ROI CROPS
# ============================================================

st.divider()
st.subheader("6. ROI Image Preview")

if page_rois:
    for name, roi in page_rois.items():
        crop_a = crop_roi(image_a, roi)
        crop_b = crop_roi(image_b, roi)

        st.markdown(f"### {name}")

        a, b = st.columns(2)

        with a:
            st.image(
                crop_a,
                caption=f"PDF A — {name}",
                use_container_width=True,
            )

        with b:
            st.image(
                crop_b,
                caption=f"PDF B — {name}",
                use_container_width=True,
            )
