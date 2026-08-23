# KYC & Citizenship Document Validator — Demo Prototype

Single-page OCR/ROI extraction tool for:
1. Nepali Citizenship Certificate
2. Siddhartha Bank KYC Form (Page 1)

## Setup

```bash
pip install -r requirements.txt
```

You also need the **Tesseract OCR engine** installed on the machine (this is
separate from the `pytesseract` Python wrapper), plus the Nepali (Devanagari)
language pack:

- **Ubuntu/Debian:**
  ```bash
  sudo apt-get install tesseract-ocr tesseract-ocr-nep
  ```
- **macOS (Homebrew):**
  ```bash
  brew install tesseract tesseract-lang
  ```
- **Windows:** install the Tesseract installer from
  https://github.com/UB-Mannheim/tesseract/wiki and select the Nepali
  language pack during setup, or drop `nep.traineddata` into the
  `tessdata` folder.

If Tesseract or the Nepali pack isn't available, the app will still run —
OCR fields will simply show an informative message instead of crashing.

## Run

```bash
streamlit run app.py
```

## Usage

1. Pick a document template in the sidebar (Citizenship Certificate or
   Siddhartha Bank KYC Form Page 1).
2. Upload a single-page PDF or image. For multi-page PDFs, pick the page.
3. Adjust the ROI sliders (as fractions of the page, 0–1) directly on the
   main screen — the red/blue/green boxes update live on the preview.
4. Click **💾 Save ROI Config** to persist coordinates to `saved_rois.json`
   so they're auto-loaded next time.
5. Click **▶️ Run OCR Extraction** to pull Devanagari text from each ROI.
6. Correct any OCR text inline, tick the verification checkboxes, and
   export the final JSON summary.

## Notes

- ROIs are stored as fractions of image width/height, so they work
  regardless of scan resolution.
- All OCR/cropping/PDF-loading code paths are wrapped in try/except and
  degrade gracefully (warnings instead of crashes) — safe for a live demo.
