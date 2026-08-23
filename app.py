import json
import re
from pathlib import Path


import fitz  # PyMuPDF

import pytesseract

from PIL import Image
import streamlit as st

from streamlit_drawable_canvas import st_canvas


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "PDF OCR Information Verifier"

ROI_FILE = Path("roi_config.json")

# If Tesseract is not in PATH, uncomment and change this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 PDF OCR Information Verifier")
st.caption(
    "Upload two PDFs, define your own ROIs, OCR them, and compare the information."
)


# ============================================================
# SESSION STATE
# ============================================================

if "rois" not in st.session_state:
    st.session_state.rois = {}

if "current_roi" not in st.session_state:
    st.session_state.current_roi = None

if "pdf_a" not in st.session_state:
    st.session_state.pdf_a = None

if "pdf_b" not in st.session_state:
    st.session_state.pdf_b = None


# ============================================================
# ROI FILE FUNCTIONS
# ============================================================

def load_roi_config():
    if not ROI_FILE.exists():
        return {}

    try:
        with open(ROI_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_roi_config():
    with open(ROI_FILE, "w", encoding="utf-8") as f:
        json.dump(
            st.session_state.rois,
            f,
            indent=4
        )


if not st.session_state.rois:
    st.session_state.rois = load_roi_config()


# ============================================================
# PDF FUNCTIONS
# ============================================================

def open_pdf(uploaded_file):
    """
    Open uploaded PDF using PyMuPDF.
    """
    if uploaded_file is None:
        return None

    pdf_bytes = uploaded_file.read()
    return fitz.open(stream=pdf_bytes, filetype="pdf")


def render_page(pdf, page_number, zoom=2.0):
    """
    Render a PDF page to a PIL image.
    """

    page = pdf.load_page(page_number)

    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    return image


# ============================================================
# IMAGE / OCR
# ============================================================

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def preprocess_image(image):
    """
    OCR preprocessing using PIL only.
    No OpenCV dependency required.
    """

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Upscale
    width, height = image.size

    image = image.resize(
        (width * 2, height * 2),
        Image.Resampling.LANCZOS
    )

    # Increase contrast
    image = ImageEnhance.Contrast(image).enhance(2.0)

    # Sharpen
    image = image.filter(
        ImageFilter.SHARPEN
    )

    # Convert to high contrast black/white
    image = ImageOps.autocontrast(image)

    return image


def normalize_text(text):
    """
    Normalize OCR result so that small formatting differences
    don't automatically count as mismatches.
    """

    text = text.lower()

    # Remove leading/trailing whitespace
    text = text.strip()

    # Replace line breaks with spaces
    text = text.replace("\n", " ")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_ocr_text(text):
    """
    More aggressive cleaning for comparison.
    """

    text = normalize_text(text)

    # Remove repeated punctuation
    text = re.sub(r"[|_~]+", " ", text)

    # Collapse whitespace again
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def ocr_image(image, language="eng"):
    """
    Run Tesseract OCR.
    """

    processed = preprocess_image(image)

    config = "--oem 3 --psm 6"

    text = pytesseract.image_to_string(
        processed,
        lang=language,
        config=config
    )

    return clean_ocr_text(text)


def ocr_with_confidence(image, language="eng"):
    """
    OCR with confidence information.
    """

    processed = preprocess_image(image)

    config = "--oem 3 --psm 6"

    data = pytesseract.image_to_data(
        processed,
        lang=language,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    words = []
    confidences = []

    for i, word in enumerate(data["text"]):

        word = word.strip()

        if not word:
            continue

        try:
            confidence = float(data["conf"][i])
        except Exception:
            confidence = -1

        words.append(word)

        if confidence >= 0:
            confidences.append(confidence)

    text = " ".join(words)

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0
    )

    return clean_ocr_text(text), average_confidence


# ============================================================
# ROI CROPPING
# ============================================================

def crop_roi(image, roi):
    """
    Crop ROI from original-resolution image.

    ROI stores normalized coordinates between 0 and 1.
    """

    width, height = image.size

    x = int(roi["x"] * width)
    y = int(roi["y"] * height)

    w = int(roi["width"] * width)
    h = int(roi["height"] * height)

    x2 = x + w
    y2 = y + h

    # Bounds checking
    x = max(0, min(x, width))
    y = max(0, min(y, height))

    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    return image.crop(
        (x, y, x2, y2)
    )


# ============================================================
# ROI MANAGEMENT
# ============================================================

def get_page_rois(page_number):
    """
    Return ROIs for a specific page.
    """

    page_key = str(page_number)

    if page_key not in st.session_state.rois:
        st.session_state.rois[page_key] = {}

    return st.session_state.rois[page_key]


def create_empty_roi():
    return {
        "x": 0.1,
        "y": 0.1,
        "width": 0.2,
        "height": 0.1
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

language = st.sidebar.selectbox(
    "OCR Language",
    [
        "eng",
        "eng+nep"
    ],
    index=0
)

st.sidebar.markdown(
    "Use `eng` for English or `eng+nep` for English + Nepali."
)

st.sidebar.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("1. Upload the two PDFs")

col1, col2 = st.columns(2)

with col1:

    uploaded_a = st.file_uploader(
        "PDF A",
        type=["pdf"],
        key="pdf_a_upload"
    )

with col2:

    uploaded_b = st.file_uploader(
        "PDF B",
        type=["pdf"],
        key="pdf_b_upload"
    )


if uploaded_a:

    st.session_state.pdf_a = open_pdf(uploaded_a)

if uploaded_b:

    st.session_state.pdf_b = open_pdf(uploaded_b)


if not uploaded_a or not uploaded_b:

    st.info(
        "Upload both PDFs to start."
    )

    st.stop()


pdf_a = st.session_state.pdf_a
pdf_b = st.session_state.pdf_b


# ============================================================
# PAGE SELECTION
# ============================================================

st.subheader("2. Choose the page")

max_pages = min(
    len(pdf_a),
    len(pdf_b)
)

page_number = st.number_input(
    "Page",
    min_value=1,
    max_value=max_pages,
    value=1,
    step=1
)

page_index = page_number - 1


# ============================================================
# RENDER PDF A
# ============================================================

image_a = render_page(
    pdf_a,
    page_index,
    zoom=2.0
)

image_b = render_page(
    pdf_b,
    page_index,
    zoom=2.0
)


# ============================================================
# ROI EDITOR
# ============================================================

st.subheader("3. ROI Editor")

page_rois = get_page_rois(page_number)

left, right = st.columns([1, 3])

with left:
    st.markdown("### ROI Manager")

    roi_names = list(page_rois.keys())

    if roi_names:
        selected_roi = st.selectbox(
            "Select ROI",
            roi_names
        )
    else:
        selected_roi = None
        st.info("Create your first ROI.")

    st.divider()

    new_roi_name = st.text_input(
        "New ROI name",
        placeholder="e.g. full_name"
    )

    if st.button("➕ Add ROI", use_container_width=True):

        if not new_roi_name.strip():
            st.error("Enter a name.")

        elif new_roi_name in page_rois:
            st.error("That ROI already exists.")

        else:
            page_rois[new_roi_name] = {
                "x": 0.10,
                "y": 0.10,
                "width": 0.20,
                "height": 0.08
            }

            save_roi_config()
            st.rerun()

    if selected_roi:

        st.divider()

        if st.button(
            "🗑️ Delete selected ROI",
            use_container_width=True
        ):
            del page_rois[selected_roi]
            save_roi_config()
            st.rerun()

with right:

    st.markdown(
        "Drag from the **top-left** to **bottom-right** of the field."
    )

    # Display image at a manageable width
    display_width = 1000

    coords = streamlit_image_coordinates(
        image_a,
        width=display_width,
        click_and_drag=True,
        cursor="crosshair",
        key=f"roi_draw_{page_number}"
    )

    if coords is not None:

        if selected_roi is None:

            st.warning(
                "Create/select an ROI name first."
            )

        else:

            x1 = coords["x1"]
            y1 = coords["y1"]
            x2 = coords["x2"]
            y2 = coords["y2"]

            # Handle dragging in either direction
            left_x = min(x1, x2)
            top_y = min(y1, y2)

            right_x = max(x1, x2)
            bottom_y = max(y1, y2)

            # Prevent tiny accidental selections
            if (
                right_x - left_x >= 5
                and bottom_y - top_y >= 5
            ):

                normalized_roi = {
                    "x": left_x / display_width,
                    "y": top_y / int(
                        image_a.height *
                        display_width /
                        image_a.width
                    ),
                    "width": (
                        right_x - left_x
                    ) / display_width,
                    "height": (
                        bottom_y - top_y
                    ) / int(
                        image_a.height *
                        display_width /
                        image_a.width
                    )
                }

                page_rois[selected_roi] = normalized_roi

                save_roi_config()

                st.success(
                    f"Saved ROI: {selected_roi}"
                )

                st.rerun()


# ============================================================
# ROI OVERVIEW
# ============================================================

st.divider()

st.subheader("4. ROI Overview")

if page_rois:

    roi_table = []

    for name, roi in page_rois.items():

        roi_table.append(
            {
                "ROI": name,
                "X": round(roi["x"], 4),
                "Y": round(roi["y"], 4),
                "Width": round(roi["width"], 4),
                "Height": round(roi["height"], 4)
            }
        )

    st.dataframe(
        roi_table,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Create at least one ROI before running OCR."
    )


# ============================================================
# OCR COMPARISON
# ============================================================

st.divider()

st.subheader("5. Compare the PDFs")


if st.button(
    "🔎 OCR + Compare",
    type="primary",
    use_container_width=True
):

    if not page_rois:

        st.error(
            "You need to create at least one ROI."
        )

        st.stop()

    results = []

    progress = st.progress(0)

    total = len(page_rois)

    for index, (name, roi) in enumerate(
        page_rois.items(),
        start=1
    ):

        crop_a = crop_roi(
            image_a,
            roi
        )

        crop_b = crop_roi(
            image_b,
            roi
        )

        text_a, confidence_a = ocr_with_confidence(
            crop_a,
            language
        )

        text_b, confidence_b = ocr_with_confidence(
            crop_b,
            language
        )

        normalized_a = normalize_text(
            text_a
        )

        normalized_b = normalize_text(
            text_b
        )

        match = (
            normalized_a == normalized_b
            and normalized_a != ""
        )

        results.append(
            {
                "ROI": name,
                "PDF A": text_a,
                "PDF B": text_b,
                "Confidence A": round(
                    confidence_a,
                    1
                ),
                "Confidence B": round(
                    confidence_b,
                    1
                ),
                "Match": match
            }
        )

        progress.progress(
            index / total
        )


    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader("Results")

    matches = sum(
        1
        for result in results
        if result["Match"]
    )

    mismatches = len(results) - matches

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total ROIs",
        len(results)
    )

    c2.metric(
        "Matching",
        matches
    )

    c3.metric(
        "Mismatching",
        mismatches
    )

    if mismatches == 0:

        st.success(
            "✅ All OCR regions match."
        )

    else:

        st.error(
            f"❌ {mismatches} ROI(s) do not match."
        )


    # ========================================================
    # INDIVIDUAL RESULTS
    # ========================================================

    for result in results:

        name = result["ROI"]

        if result["Match"]:

            with st.expander(
                f"✅ {name}",
                expanded=False
            ):

                st.write(
                    f"**PDF A:** {result['PDF A']}"
                )

                st.write(
                    f"**PDF B:** {result['PDF B']}"
                )

                st.write(
                    f"Confidence A: {result['Confidence A']}"
                )

                st.write(
                    f"Confidence B: {result['Confidence B']}"
                )

        else:

            with st.expander(
                f"❌ {name}",
                expanded=True
            ):

                a, b = st.columns(2)

                with a:

                    st.markdown("### PDF A")

                    st.error(
                        result["PDF A"]
                        if result["PDF A"]
                        else "[No OCR text]"
                    )

                    st.caption(
                        f"Confidence: {result['Confidence A']}"
                    )

                with b:

                    st.markdown("### PDF B")

                    st.error(
                        result["PDF B"]
                        if result["PDF B"]
                        else "[No OCR text]"
                    )

                    st.caption(
                        f"Confidence: {result['Confidence B']}"
                    )


# ============================================================
# DEBUG PREVIEW
# ============================================================

st.divider()

st.subheader("6. ROI Preview")

if page_rois:

    for name, roi in page_rois.items():

        crop_a = crop_roi(
            image_a,
            roi
        )

        crop_b = crop_roi(
            image_b,
            roi
        )

        st.markdown(
            f"### {name}"
        )

        a, b = st.columns(2)

        with a:

            st.image(
                crop_a,
                caption=f"PDF A — {name}",
                use_container_width=True
            )

        with b:

            st.image(
                crop_b,
                caption=f"PDF B — {name}",
                use_container_width=True
            )
