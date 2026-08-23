"""
KYC & Citizenship Document Validator — Bank Demo Prototype
============================================================
Built for: Siddhartha Bank KYC Form (Page 1) & Nepali Citizenship Certificate
Stack: Streamlit + PyMuPDF (fitz) + OpenCV + Pillow + pytesseract

Run with:  streamlit run app.py
"""

import io
import json
import os
from datetime import datetime

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------
# CONSTANTS & CONFIG
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="KYC & Citizenship Document Validator",
    page_icon="🏦",
    layout="wide",
)

ROI_CONFIG_PATH = "saved_rois.json"

TEMPLATES = {
    "Citizenship Certificate": [
        "Full Name (नाम थर)",
        "Citizenship Number (ना. प्र. नं.)",
        "Date of Birth (जन्म मिति)",
        "Permanent Address (स्थायी बासस्थान)",
    ],
    "Siddhartha Bank KYC Form (Page 1)": [
        "Applicant Name (निवेदकको नाम)",
        "Date of Birth (जन्म मिति)",
        "Citizenship No. (नागरिकता नं.)",
        "Issue District (जारी भएको जिल्ला)",
    ],
}

# Default ROIs expressed as FRACTIONS of image width/height (0.0 - 1.0).
# Fractional coordinates make the ROI boxes automatically scale to any
# uploaded document resolution, which removes the #1 source of
# out-of-bounds crashes when moving between scans of different sizes.
DEFAULT_ROIS = {
    "Citizenship Certificate": {
        "Full Name (नाम थर)": {"x": 0.28, "y": 0.20, "w": 0.60, "h": 0.06},
        "Citizenship Number (ना. प्र. नं.)": {"x": 0.28, "y": 0.28, "w": 0.60, "h": 0.06},
        "Date of Birth (जन्म मिति)": {"x": 0.28, "y": 0.36, "w": 0.60, "h": 0.06},
        "Permanent Address (स्थायी बासस्थान)": {"x": 0.28, "y": 0.44, "w": 0.60, "h": 0.09},
    },
    "Siddhartha Bank KYC Form (Page 1)": {
        "Applicant Name (निवेदकको नाम)": {"x": 0.30, "y": 0.15, "w": 0.62, "h": 0.05},
        "Date of Birth (जन्म मिति)": {"x": 0.30, "y": 0.22, "w": 0.35, "h": 0.05},
        "Citizenship No. (नागरिकता नं.)": {"x": 0.30, "y": 0.29, "w": 0.45, "h": 0.05},
        "Issue District (जारी भएको जिल्ला)": {"x": 0.30, "y": 0.36, "w": 0.45, "h": 0.05},
    },
}

BOX_COLORS = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (233, 111, 45),
    (106, 76, 156),
    (38, 70, 83),
]

CHECK_ITEMS = [
    ("legible", "☑️ Document Legible"),
    ("details_match", "☑️ Details Match Bank Records"),
    ("kyc_approved", "☑️ KYC Approved"),
]


# ----------------------------------------------------------------------------
# ROI CONFIG PERSISTENCE
# ----------------------------------------------------------------------------

def load_roi_config() -> dict:
    """Load saved ROI config from disk, falling back to defaults on any issue.
    Guarantees every template/field key exists so the UI never KeyErrors,
    even if the JSON on disk is partial, stale, or corrupted."""
    config = json.loads(json.dumps(DEFAULT_ROIS))  # deep copy of defaults

    if os.path.exists(ROI_CONFIG_PATH):
        try:
            with open(ROI_CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for template, fields in TEMPLATES.items():
                if template in saved and isinstance(saved[template], dict):
                    for field in fields:
                        roi = saved[template].get(field)
                        if isinstance(roi, dict) and all(k in roi for k in ("x", "y", "w", "h")):
                            config[template][field] = {
                                "x": float(roi["x"]),
                                "y": float(roi["y"]),
                                "w": float(roi["w"]),
                                "h": float(roi["h"]),
                            }
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            st.sidebar.warning(f"⚠️ Could not read saved ROI file, using defaults. ({e})")

    return config


def save_roi_config(rois: dict) -> bool:
    try:
        with open(ROI_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(rois, f, ensure_ascii=False, indent=2)
        return True
    except OSError as e:
        st.error(f"❌ Could not save ROI config: {e}")
        return False


# ----------------------------------------------------------------------------
# SESSION STATE INITIALISATION
# ----------------------------------------------------------------------------

def init_session_state():
    if "rois" not in st.session_state:
        st.session_state.rois = load_roi_config()
    if "doc_type" not in st.session_state:
        st.session_state.doc_type = "Citizenship Certificate"
    if "working_image" not in st.session_state:
        st.session_state.working_image = None
    if "page_num" not in st.session_state:
        st.session_state.page_num = 1
    if "extracted" not in st.session_state:
        st.session_state.extracted = {}
    if "edited" not in st.session_state:
        st.session_state.edited = {}
    if "checks" not in st.session_state:
        st.session_state.checks = {key: False for key, _ in CHECK_ITEMS}
    if "ocr_lang" not in st.session_state:
        st.session_state.ocr_lang = "nep+eng"
    if "last_upload_name" not in st.session_state:
        st.session_state.last_upload_name = None
    # Version counters force Streamlit to mint fresh widget keys whenever
    # we update a value *programmatically* (OCR results, ROI reset).
    # Streamlit widgets ignore a new `value=` once their key already exists
    # in session_state, so bumping the version is what makes those
    # programmatic updates actually show up on screen.
    if "extract_version" not in st.session_state:
        st.session_state.extract_version = 0
    if "roi_version" not in st.session_state:
        st.session_state.roi_version = 0


# ----------------------------------------------------------------------------
# DOCUMENT LOADING (PDF / IMAGE) — strictly single page at a time
# ----------------------------------------------------------------------------

def render_pdf_page(file_bytes: bytes, page_index: int, dpi: int = 220) -> Image.Image:
    """Render a single PDF page (0-indexed) to a PIL RGB Image."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError("Page index out of range.")
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        return img
    finally:
        doc.close()


def get_pdf_page_count(file_bytes: bytes) -> int:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()


# ----------------------------------------------------------------------------
# ROI GEOMETRY HELPERS
# ----------------------------------------------------------------------------

def fractional_to_pixels(roi: dict, img_w: int, img_h: int):
    """Convert a fractional ROI {x,y,w,h} into a safely clamped pixel box."""
    x = int(round(max(0.0, min(0.99, float(roi.get("x", 0.0)))) * img_w))
    y = int(round(max(0.0, min(0.99, float(roi.get("y", 0.0)))) * img_h))
    w = int(round(max(0.01, min(1.0, float(roi.get("w", 0.1)))) * img_w))
    h = int(round(max(0.01, min(1.0, float(roi.get("h", 0.1)))) * img_h))

    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def draw_roi_overlay(base_image: Image.Image, fields: list, rois: dict) -> Image.Image:
    annotated = base_image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    img_w, img_h = annotated.size

    for i, field in enumerate(fields):
        roi = rois.get(field)
        if not roi:
            continue
        try:
            x, y, w, h = fractional_to_pixels(roi, img_w, img_h)
            color = BOX_COLORS[i % len(BOX_COLORS)]
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            label = str(i + 1)
            label_bg = [x, max(0, y - 18), x + 18, y]
            draw.rectangle(label_bg, fill=color)
            draw.text((x + 4, max(0, y - 17)), label, fill=(255, 255, 255), font=font)
        except Exception as e:
            st.warning(f"⚠️ Could not draw ROI for '{field}': {e}")

    return annotated


def crop_roi(image: Image.Image, roi: dict) -> Image.Image:
    img_w, img_h = image.size
    x, y, w, h = fractional_to_pixels(roi, img_w, img_h)
    return image.crop((x, y, x + w, y + h))


# ----------------------------------------------------------------------------
# OCR PIPELINE
# ----------------------------------------------------------------------------

def preprocess_for_ocr(pil_crop: Image.Image) -> np.ndarray:
    """OpenCV preprocessing to boost Devanagari OCR accuracy on small crops."""
    try:
        rgb = np.array(pil_crop.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Upscale small ROI crops — Tesseract performs noticeably better
        # on Devanagari script above ~300dpi-equivalent glyph size.
        gray = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)

        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
        return thresh
    except Exception:
        # Absolute fallback — never let preprocessing crash the pipeline.
        try:
            return np.array(pil_crop.convert("L"))
        except Exception:
            return np.array(pil_crop)


def run_ocr_on_roi(image: Image.Image, roi: dict, lang: str) -> str:
    """Crop + preprocess + OCR a single field. Never raises — returns ''
    and lets the caller decide how to surface the problem."""
    try:
        crop = crop_roi(image, roi)
        processed = preprocess_for_ocr(crop)
        config = "--oem 3 --psm 7"
        text = pytesseract.image_to_string(processed, lang=lang, config=config)
        return text.strip()
    except pytesseract.pytesseract.TesseractNotFoundError:
        raise
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.title("🏦 Validator Controls")

    doc_type = st.sidebar.radio(
        "Document Type",
        options=list(TEMPLATES.keys()),
        index=list(TEMPLATES.keys()).index(st.session_state.doc_type),
        key="doc_type_radio",
    )
    if doc_type != st.session_state.doc_type:
        st.session_state.doc_type = doc_type
        # Reset per-document working state on template switch so stale
        # text/checks from a different template never leak across.
        st.session_state.extracted = {}
        st.session_state.edited = {}

    st.sidebar.divider()
    st.sidebar.subheader("🔤 OCR Settings")
    st.session_state.ocr_lang = st.sidebar.text_input(
        "Tesseract language code",
        value=st.session_state.ocr_lang,
        help="e.g. 'nep+eng' for Devanagari + Latin, or 'nep' for Devanagari only.",
    )
    tess_cmd = st.sidebar.text_input(
        "Tesseract executable path (optional)",
        value="",
        placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        help="Only needed on Windows machines where tesseract isn't on PATH.",
    )
    if tess_cmd.strip():
        pytesseract.pytesseract.tesseract_cmd = tess_cmd.strip()

    st.sidebar.divider()
    st.sidebar.subheader("📐 ROI Configuration")

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("💾 Save ROI Config", use_container_width=True):
            if save_roi_config(st.session_state.rois):
                st.sidebar.success("Saved to saved_rois.json")
    with col_b:
        if st.button("↺ Reset Template", use_container_width=True):
            st.session_state.rois[st.session_state.doc_type] = json.loads(
                json.dumps(DEFAULT_ROIS[st.session_state.doc_type])
            )
            st.session_state.roi_version += 1  # force sliders to re-init from defaults
            st.sidebar.info("ROIs reset to defaults for this template.")
            st.rerun()

    st.sidebar.caption(
        "ROI positions persist automatically to a local JSON file so the "
        "next demo session reloads your tuned coordinates."
    )


# ----------------------------------------------------------------------------
# DOCUMENT UPLOAD + PAGE SELECTION
# ----------------------------------------------------------------------------

def render_uploader():
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a single-page PDF or image (PNG/JPG) of the document",
        type=["pdf", "png", "jpg", "jpeg"],
        key="file_uploader",
    )

    if uploaded_file is None:
        st.session_state.working_image = None
        st.session_state.last_upload_name = None
        st.info("👆 Upload a Citizenship Certificate or KYC Form page to begin.")
        return

    try:
        file_bytes = uploaded_file.getvalue()
        is_pdf = uploaded_file.name.lower().endswith(".pdf")

        if is_pdf:
            page_count = get_pdf_page_count(file_bytes)
            if page_count < 1:
                st.error("❌ The uploaded PDF appears to have no pages.")
                st.session_state.working_image = None
                return

            page_options = list(range(1, page_count + 1))
            selected_page = st.selectbox(
                "Select page to process (single page at a time)",
                options=page_options,
                index=min(st.session_state.page_num - 1, page_count - 1),
                key="pdf_page_selector",
            )
            st.session_state.page_num = selected_page
            st.session_state.working_image = render_pdf_page(file_bytes, selected_page - 1)
        else:
            img = Image.open(io.BytesIO(file_bytes))
            img = img.convert("RGB")
            st.session_state.working_image = img
            st.session_state.page_num = 1

        st.session_state.last_upload_name = uploaded_file.name

    except fitz.FileDataError:
        st.error("❌ This PDF could not be read — it may be corrupted or password-protected.")
        st.session_state.working_image = None
    except Exception as e:
        st.error(f"❌ Could not load document: {e}")
        st.session_state.working_image = None


# ----------------------------------------------------------------------------
# MAIN TWO-COLUMN WORKSPACE
# ----------------------------------------------------------------------------

def render_left_column(fields: list):
    st.markdown("#### 🖼️ Document Preview")
    image = st.session_state.working_image
    if image is None:
        st.warning("⚠️ No document loaded yet.")
        return

    try:
        rois = st.session_state.rois[st.session_state.doc_type]
        annotated = draw_roi_overlay(image, fields, rois)
        st.image(annotated, use_container_width=True, caption=f"Page {st.session_state.page_num}")
    except Exception as e:
        st.error(f"❌ Could not render preview overlay: {e}")
        st.image(image, use_container_width=True)

    legend = "  ".join(
        f"**{i+1}.** {field}" for i, field in enumerate(fields)
    )
    st.caption(legend)


def render_field_roi_editor(field: str, index: int):
    doc_type = st.session_state.doc_type
    roi = st.session_state.rois[doc_type].get(field, {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.05})

    key_prefix = f"roi_{doc_type}_{index}_{st.session_state.roi_version}"
    c1, c2 = st.columns(2)
    with c1:
        x = st.slider("x (left)", 0.0, 0.99, float(roi["x"]), 0.01, key=f"{key_prefix}_x")
        w = st.slider("w (width)", 0.01, 1.0, float(roi["w"]), 0.01, key=f"{key_prefix}_w")
    with c2:
        y = st.slider("y (top)", 0.0, 0.99, float(roi["y"]), 0.01, key=f"{key_prefix}_y")
        h = st.slider("h (height)", 0.01, 1.0, float(roi["h"]), 0.01, key=f"{key_prefix}_h")

    st.session_state.rois[doc_type][field] = {"x": x, "y": y, "w": w, "h": h}


def render_right_column(fields: list):
    st.markdown("#### 📝 Extracted Fields & Verification")

    image = st.session_state.working_image

    extract_col, _ = st.columns([1, 2])
    with extract_col:
        run_clicked = st.button("🔍 Extract Text from ROIs", use_container_width=True, type="primary")

    if run_clicked:
        if image is None:
            st.warning("⚠️ Upload a document before running OCR.")
        else:
            try:
                rois = st.session_state.rois[st.session_state.doc_type]
                with st.spinner("Running OCR on selected regions..."):
                    for field in fields:
                        try:
                            text = run_ocr_on_roi(image, rois[field], st.session_state.ocr_lang)
                        except pytesseract.pytesseract.TesseractNotFoundError:
                            st.error(
                                "❌ Tesseract OCR engine not found. Install Tesseract "
                                "(with the Devanagari 'nep' language pack) and/or set "
                                "the executable path in the sidebar."
                            )
                            text = ""
                        st.session_state.extracted[field] = text
                        st.session_state.edited[field] = text
                st.session_state.extract_version += 1  # force text widgets to show new OCR text
                st.success("✅ OCR extraction complete. Review and correct below.")
            except Exception as e:
                st.error(f"❌ OCR pipeline error: {e}")

    for i, field in enumerate(fields):
        with st.expander(f"**{i + 1}. {field}**", expanded=(i == 0)):
            render_field_roi_editor(field, i)

            ver = st.session_state.extract_version
            raw_text = st.session_state.extracted.get(field, "")
            st.text_input(
                "OCR Raw Output (read-only)",
                value=raw_text,
                disabled=True,
                key=f"raw_{st.session_state.doc_type}_{i}_{ver}",
            )

            current_edit = st.session_state.edited.get(field, raw_text)
            edited_value = st.text_area(
                "Verified / Corrected Value",
                value=current_edit,
                key=f"edit_{st.session_state.doc_type}_{i}_{ver}",
                height=68,
            )
            st.session_state.edited[field] = edited_value

    st.divider()
    render_verification_checkboxes()
    st.divider()
    render_summary_export(fields)


def render_verification_checkboxes():
    st.markdown("#### ✅ Verification Checklist")
    cols = st.columns(len(CHECK_ITEMS))
    for col, (key, label) in zip(cols, CHECK_ITEMS):
        with col:
            st.session_state.checks[key] = st.checkbox(
                label, value=st.session_state.checks.get(key, False), key=f"chk_{key}"
            )


def render_summary_export(fields: list):
    st.markdown("#### 📄 Verification Summary")

    summary = {
        "document_type": st.session_state.doc_type,
        "page_number": st.session_state.page_num,
        "source_file": st.session_state.last_upload_name,
        "extracted_fields": {field: st.session_state.edited.get(field, "") for field in fields},
        "verification": {
            label: st.session_state.checks.get(key, False) for key, label in CHECK_ITEMS
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    st.json(summary, expanded=False)

    try:
        json_bytes = json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download JSON Summary",
            data=json_bytes,
            file_name=f"kyc_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"❌ Could not prepare download: {e}")


# ----------------------------------------------------------------------------
# APP ENTRY POINT
# ----------------------------------------------------------------------------

def main():
    init_session_state()
    render_sidebar()

    st.title("🏦 KYC & Citizenship Document Validator")
    st.caption(
        "Prototype for Siddhartha Bank — Citizenship Certificate & KYC Form (Page 1) verification"
    )

    render_uploader()
    st.divider()

    fields = TEMPLATES[st.session_state.doc_type]

    left, right = st.columns([1, 1])
    with left:
        render_left_column(fields)
    with right:
        render_right_column(fields)


if __name__ == "__main__":
    main()
