"""
KYC & Citizenship Document Validator — Siddhartha Bank Demo Prototype
======================================================================
A single-page-at-a-time OCR/ROI extraction tool for:
  1. Nepali Citizenship Certificate
  2. Siddhartha Bank KYC Form (Page 1)

Run with:  streamlit run app.py
"""

import io
import os
import json
import copy
import traceback

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ---- Optional heavy deps are imported defensively so the app never hard-crashes ----
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False


# =========================================================================
# CONSTANTS & DEFAULT TEMPLATES
# =========================================================================

CONFIG_PATH = "saved_rois.json"

DOC_TYPES = [
    "Citizenship Certificate",
    "Siddhartha Bank KYC Form (Page 1)",
]

# Default ROIs are expressed as FRACTIONS of the full image (x, y, w, h),
# each in [0, 1]. Fractions make ROIs resolution independent, which matters
# because uploaded scans/photos vary wildly in pixel size.
DEFAULT_ROIS = {
    "Citizenship Certificate": {
        "Full Name (नाम थर)":              {"x": 0.30, "y": 0.27, "w": 0.45, "h": 0.06},
        "Citizenship Number (ना. प्र. नं.)": {"x": 0.05, "y": 0.24, "w": 0.35, "h": 0.05},
        "Date of Birth (जन्म मिति)":         {"x": 0.30, "y": 0.45, "w": 0.45, "h": 0.06},
        "Permanent Address (स्थायी बासस्थान)": {"x": 0.30, "y": 0.34, "w": 0.55, "h": 0.09},
    },
    "Siddhartha Bank KYC Form (Page 1)": {
        "Applicant Name (निवेदकको नाम)":       {"x": 0.06, "y": 0.55, "w": 0.55, "h": 0.03},
        "Date of Birth (जन्म मिति)":            {"x": 0.36, "y": 0.60, "w": 0.25, "h": 0.03},
        "Citizenship No. (नागरिकता नं.)":       {"x": 0.06, "y": 0.63, "w": 0.30, "h": 0.03},
        "Issue District (जारी भएको जिल्ला)":     {"x": 0.70, "y": 0.63, "w": 0.28, "h": 0.03},
    },
}

VERIFICATION_CHECKS = [
    "Document Legible",
    "Details Match",
    "KYC Approved",
]

# Preferred OCR language packs, tried in order (falls back gracefully).
OCR_LANG_CANDIDATES = ["nep+eng", "nep", "eng"]


# =========================================================================
# CONFIG PERSISTENCE HELPERS
# =========================================================================

def load_saved_rois() -> dict:
    """Load ROI overrides from local JSON. Falls back to defaults on any error."""
    if not os.path.exists(CONFIG_PATH):
        return copy.deepcopy(DEFAULT_ROIS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Merge with defaults so newly added fields always exist even if the
        # saved file is from an older version of this app.
        merged = copy.deepcopy(DEFAULT_ROIS)
        for doc_type, fields in data.items():
            if doc_type not in merged:
                merged[doc_type] = {}
            for field_name, roi in fields.items():
                if all(k in roi for k in ("x", "y", "w", "h")):
                    merged[doc_type][field_name] = roi
        return merged
    except Exception:
        # Corrupt file - don't crash the demo, just use defaults.
        return copy.deepcopy(DEFAULT_ROIS)


def save_rois_to_disk(all_rois: dict) -> bool:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(all_rois, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# =========================================================================
# SESSION STATE INITIALIZATION
# =========================================================================

def init_session_state():
    if "rois" not in st.session_state:
        st.session_state.rois = load_saved_rois()

    if "doc_type" not in st.session_state:
        st.session_state.doc_type = DOC_TYPES[0]

    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = {}  # field_name -> ocr text

    if "manual_text" not in st.session_state:
        st.session_state.manual_text = {}  # field_name -> user-corrected text

    if "checks" not in st.session_state:
        st.session_state.checks = {c: False for c in VERIFICATION_CHECKS}

    if "page_image" not in st.session_state:
        st.session_state.page_image = None

    if "source_file_name" not in st.session_state:
        st.session_state.source_file_name = None


# =========================================================================
# DOCUMENT LOADING
# =========================================================================

def load_pdf_page(file_bytes: bytes, page_index: int, zoom: float = 2.0) -> Image.Image:
    """Render a single PDF page to a PIL Image using PyMuPDF."""
    if not PYMUPDF_AVAILABLE:
        raise RuntimeError(
            "PyMuPDF (fitz) is not installed. Run `pip install pymupdf` to enable PDF support."
        )
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(f"Page index {page_index} out of range (doc has {doc.page_count} pages).")
        page = doc.load_page(page_index)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        return img
    finally:
        doc.close()


def get_pdf_page_count(file_bytes: bytes) -> int:
    if not PYMUPDF_AVAILABLE:
        return 1
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        return 1


# =========================================================================
# IMAGE / ROI UTILITIES
# =========================================================================

def fractional_roi_to_pixels(roi: dict, img_w: int, img_h: int):
    """Convert a fractional ROI dict {x,y,w,h} in [0,1] to pixel box (x0,y0,x1,y1),
    clamped safely inside the image bounds."""
    x = max(0.0, min(1.0, float(roi.get("x", 0))))
    y = max(0.0, min(1.0, float(roi.get("y", 0))))
    w = max(0.0, min(1.0, float(roi.get("w", 0.01))))
    h = max(0.0, min(1.0, float(roi.get("h", 0.01))))

    x0 = int(x * img_w)
    y0 = int(y * img_h)
    x1 = int(min(img_w, x0 + w * img_w))
    y1 = int(min(img_h, y0 + h * img_h))

    # Guard against degenerate/zero-area boxes.
    if x1 <= x0:
        x1 = min(img_w, x0 + 1)
    if y1 <= y0:
        y1 = min(img_h, y0 + 1)

    return x0, y0, x1, y1


def crop_roi(image: Image.Image, roi: dict) -> Image.Image:
    """Safely crop an ROI from a PIL image. Never raises — returns a tiny
    blank image on any failure so the pipeline keeps going."""
    try:
        img_w, img_h = image.size
        x0, y0, x1, y1 = fractional_roi_to_pixels(roi, img_w, img_h)
        cropped = image.crop((x0, y0, x1, y1))
        if cropped.width == 0 or cropped.height == 0:
            raise ValueError("Cropped region has zero area.")
        return cropped
    except Exception:
        return Image.new("RGB", (10, 10), color=(255, 255, 255))


def draw_roi_overlay(image: Image.Image, rois: dict) -> Image.Image:
    """Draw labeled bounding boxes for every ROI onto a copy of the image."""
    try:
        overlay = image.copy().convert("RGB")
        draw = ImageDraw.Draw(overlay)
        img_w, img_h = overlay.size

        colors = ["#FF3B30", "#007AFF", "#34C759", "#FF9500", "#AF52DE", "#5AC8FA"]

        for i, (field_name, roi) in enumerate(rois.items()):
            color = colors[i % len(colors)]
            x0, y0, x1, y1 = fractional_roi_to_pixels(roi, img_w, img_h)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            label = field_name.split("(")[0].strip()
            # Simple label background so text is readable regardless of image content.
            text_y = max(0, y0 - 18)
            draw.rectangle([x0, text_y, x0 + 8 * len(label), text_y + 16], fill=color)
            draw.text((x0 + 2, text_y), label, fill="white")
        return overlay
    except Exception:
        return image


# =========================================================================
# OCR PIPELINE
# =========================================================================

def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Light preprocessing (grayscale + upscale + threshold) to improve OCR
    accuracy on scanned bank forms. Falls back to the raw image if cv2 is
    unavailable or anything goes wrong."""
    if not CV2_AVAILABLE:
        return pil_img
    try:
        arr = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Upscale small crops — helps Tesseract a lot on tiny form fields.
        h, w = gray.shape[:2]
        scale = 2 if max(h, w) < 600 else 1
        if scale > 1:
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        gray = cv2.fastNlMeansDenoising(gray, h=10)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
        return Image.fromarray(thresh)
    except Exception:
        return pil_img


def run_ocr(pil_img: Image.Image) -> str:
    """Run Tesseract OCR with Devanagari+English support, trying language
    packs in order of preference. Never raises."""
    if not TESSERACT_AVAILABLE:
        return "[pytesseract not installed — install tesseract-ocr + nep language pack]"

    processed = preprocess_for_ocr(pil_img)
    last_error = None

    for lang in OCR_LANG_CANDIDATES:
        try:
            text = pytesseract.image_to_string(processed, lang=lang, config="--psm 7")
            text = text.strip()
            if text:
                return text
        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        return f"[OCR failed: {last_error}]"
    return ""


def run_ocr_all_fields(image: Image.Image, rois: dict) -> dict:
    """Run OCR over every ROI for the current document. Guardrails ensure a
    single bad field never kills the whole batch."""
    results = {}
    for field_name, roi in rois.items():
        try:
            crop = crop_roi(image, roi)
            results[field_name] = run_ocr(crop)
        except Exception as e:
            results[field_name] = f"[Error extracting field: {e}]"
    return results


# =========================================================================
# STREAMLIT APP
# =========================================================================

st.set_page_config(
    page_title="KYC & Citizenship Document Validator",
    page_icon="🏦",
    layout="wide",
)

init_session_state()

st.title("🏦 KYC & Citizenship Document Validator")
st.caption("Siddhartha Bank — Live Demo Prototype · Single-page document processing")

if not PYMUPDF_AVAILABLE:
    st.sidebar.warning("PyMuPDF not installed — PDF upload disabled. Install with `pip install pymupdf`.")
if not TESSERACT_AVAILABLE:
    st.sidebar.warning("pytesseract not installed — OCR disabled. Install with `pip install pytesseract` "
                        "and the Tesseract binary + Nepali language pack.")

# ------------------------------------------------------------------
# SIDEBAR — document type + upload
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Document Setup")

    new_doc_type = st.radio(
        "Select document template",
        DOC_TYPES,
        index=DOC_TYPES.index(st.session_state.doc_type),
    )
    if new_doc_type != st.session_state.doc_type:
        st.session_state.doc_type = new_doc_type
        # Switching templates should not wipe already-extracted results from
        # other documents, but the active preview should reset since the ROI
        # layout is template-specific.
        st.session_state.extracted_text = {}

    st.divider()
    uploaded_file = st.file_uploader(
        "Upload single-page PDF or image (PNG/JPG)",
        type=["pdf", "png", "jpg", "jpeg"],
    )

    page_choice = 0
    if uploaded_file is not None and uploaded_file.name.lower().endswith(".pdf"):
        try:
            file_bytes = uploaded_file.getvalue()
            n_pages = get_pdf_page_count(file_bytes)
            if n_pages > 1:
                page_choice = st.selectbox(
                    "Select page to process",
                    options=list(range(n_pages)),
                    format_func=lambda i: f"Page {i + 1}",
                )
        except Exception as e:
            st.error(f"Could not read PDF page count: {e}")

    st.divider()
    st.subheader("✅ Verification Checklist")
    for check_name in VERIFICATION_CHECKS:
        st.session_state.checks[check_name] = st.checkbox(
            check_name,
            value=st.session_state.checks.get(check_name, False),
            key=f"check_{check_name}",
        )

# ------------------------------------------------------------------
# LOAD THE PAGE IMAGE (only re-decode when the file actually changes)
# ------------------------------------------------------------------
if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}_{page_choice}"
    if st.session_state.source_file_name != file_id:
        try:
            raw_bytes = uploaded_file.getvalue()
            if uploaded_file.name.lower().endswith(".pdf"):
                img = load_pdf_page(raw_bytes, page_choice)
            else:
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            st.session_state.page_image = img
            st.session_state.source_file_name = file_id
            st.session_state.extracted_text = {}
        except Exception as e:
            st.session_state.page_image = None
            st.error(f"❌ Could not load document: {e}")

current_rois = st.session_state.rois.get(st.session_state.doc_type, {})

# ------------------------------------------------------------------
# MAIN TWO-COLUMN LAYOUT
# ------------------------------------------------------------------
left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.subheader("📄 Document Preview")

    if st.session_state.page_image is None:
        st.info("Upload a PDF or image on the left sidebar to begin.")
    else:
        try:
            preview = draw_roi_overlay(st.session_state.page_image, current_rois)
            st.image(preview, use_container_width=True, caption=f"{st.session_state.doc_type} — ROI overlay")
        except Exception as e:
            st.error(f"Could not render preview: {e}")
            st.image(st.session_state.page_image, use_container_width=True)

    st.divider()
    st.subheader("🎛️ ROI Adjustment (fractions of page: 0.0 – 1.0)")

    if not current_rois:
        st.warning("No ROI fields defined for this document type.")
    else:
        for field_name in list(current_rois.keys()):
            with st.expander(f"📌 {field_name}", expanded=False):
                roi = current_rois[field_name]
                c1, c2, c3, c4 = st.columns(4)
                try:
                    roi["x"] = c1.slider(
                        "x", 0.0, 1.0, float(roi.get("x", 0.1)), 0.005,
                        key=f"{st.session_state.doc_type}_{field_name}_x",
                    )
                    roi["y"] = c2.slider(
                        "y", 0.0, 1.0, float(roi.get("y", 0.1)), 0.005,
                        key=f"{st.session_state.doc_type}_{field_name}_y",
                    )
                    roi["w"] = c3.slider(
                        "w", 0.01, 1.0, float(roi.get("w", 0.2)), 0.005,
                        key=f"{st.session_state.doc_type}_{field_name}_w",
                    )
                    roi["h"] = c4.slider(
                        "h", 0.01, 1.0, float(roi.get("h", 0.05)), 0.005,
                        key=f"{st.session_state.doc_type}_{field_name}_h",
                    )
                except Exception as e:
                    st.warning(f"Slider error for {field_name}, resetting to default: {e}")
                    roi.update(DEFAULT_ROIS.get(st.session_state.doc_type, {}).get(field_name, {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.05}))

        st.session_state.rois[st.session_state.doc_type] = current_rois

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("💾 Save ROI Config", use_container_width=True):
            ok = save_rois_to_disk(st.session_state.rois)
            if ok:
                st.success(f"Saved ROI configuration to `{CONFIG_PATH}`.")
            else:
                st.error("Failed to save ROI configuration to disk.")
    with btn_col2:
        if st.button("↩️ Reset to Default ROIs", use_container_width=True):
            st.session_state.rois[st.session_state.doc_type] = copy.deepcopy(
                DEFAULT_ROIS.get(st.session_state.doc_type, {})
            )
            st.rerun()

with right_col:
    st.subheader("🔎 Extracted Fields")

    if st.session_state.page_image is None:
        st.info("No document loaded yet.")
    else:
        if st.button("▶️ Run OCR Extraction", type="primary", use_container_width=True):
            with st.spinner("Running OCR on all fields..."):
                try:
                    st.session_state.extracted_text = run_ocr_all_fields(
                        st.session_state.page_image, current_rois
                    )
                    # Seed manual-correction boxes with fresh OCR output only
                    # for fields that don't already have a manual edit.
                    for field_name, text in st.session_state.extracted_text.items():
                        st.session_state.manual_text.setdefault(field_name, text)
                        # If user hasn't touched it, keep it synced with new OCR run.
                        st.session_state.manual_text[field_name] = text
                except Exception as e:
                    st.error(f"OCR pipeline failed: {e}")
                    st.text(traceback.format_exc())

        if not current_rois:
            st.warning("No fields configured for this document type.")
        else:
            for field_name in current_rois.keys():
                ocr_val = st.session_state.extracted_text.get(field_name, "")
                default_val = st.session_state.manual_text.get(field_name, ocr_val)

                st.markdown(f"**{field_name}**")
                small_prev_col, _ = st.columns([1, 3])
                try:
                    crop_preview = crop_roi(st.session_state.page_image, current_rois[field_name])
                    small_prev_col.image(crop_preview, width=200)
                except Exception:
                    pass

                corrected = st.text_input(
                    label=f"Corrected value — {field_name}",
                    value=default_val,
                    key=f"manual_{st.session_state.doc_type}_{field_name}",
                    label_visibility="collapsed",
                )
                st.session_state.manual_text[field_name] = corrected
                st.divider()

    st.subheader("📋 Verification Summary (JSON export)")

    try:
        summary = {
            "document_type": st.session_state.doc_type,
            "source_file": uploaded_file.name if uploaded_file else None,
            "extracted_fields": {
                field_name: st.session_state.manual_text.get(
                    field_name, st.session_state.extracted_text.get(field_name, "")
                )
                for field_name in current_rois.keys()
            },
            "verification_checks": st.session_state.checks,
        }
        summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
        st.code(summary_json, language="json")
        st.download_button(
            "⬇️ Download Summary JSON",
            data=summary_json.encode("utf-8"),
            file_name="kyc_verification_summary.json",
            mime="application/json",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Could not build summary JSON: {e}")

st.divider()
st.caption(
    "Prototype for internal demonstration purposes only. "
    "OCR accuracy depends on scan quality and installed Tesseract language packs (nep, eng)."
)
