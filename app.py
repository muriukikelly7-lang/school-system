"""Streamlit interface for scanning school exam cover pages."""

from __future__ import annotations

import csv
from io import BytesIO
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
STUDENTS_CSV = BASE_DIR / "students_db.csv"
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024
MAX_OCR_DIMENSION = 2400
MAX_OCR_UPSCALE = 2.5
SAMPLE_STUDENTS = [
    {"Admission_No": "ADM-101", "Student_Name": "John Doe", "Class": "Form 3"},
    {"Admission_No": "ADM-102", "Student_Name": "Jane Smith", "Class": "Form 3"},
    {"Admission_No": "ADM-103", "Student_Name": "Alex Otieno", "Class": "Form 3"},
]


def create_students_database_if_missing() -> None:
    """Create the example database when it does not exist; preserve user data."""
    if STUDENTS_CSV.exists():
        return

    with STUDENTS_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["Admission_No", "Student_Name", "Class"]
        )
        writer.writeheader()
        writer.writerows(SAMPLE_STUDENTS)


def load_students() -> dict[str, dict[str, str]]:
    """Load student records keyed by normalized admission number."""
    create_students_database_if_missing()
    required_columns = {"Admission_No", "Student_Name", "Class"}
    lookup: dict[str, dict[str, str]] = {}

    with STUDENTS_CSV.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{STUDENTS_CSV.name} is missing required columns: {missing}")

        for row in reader:
            admission_no = normalize_admission_number(row.get("Admission_No") or "")
            if admission_no:
                lookup[admission_no] = {
                    "Student_Name": (row.get("Student_Name") or "").strip(),
                    "Class": (row.get("Class") or "").strip(),
                }
    return lookup


def normalize_admission_number(value: str) -> str | None:
    """Normalize an ADM-### or ADM### value, returning None if invalid."""
    match = re.fullmatch(r"\s*ADM\s*-?\s*(\d{3})\s*", value, re.IGNORECASE)
    return f"ADM-{match.group(1)}" if match else None


def find_tesseract() -> str:
    """Find Tesseract on PATH or in its standard Windows installation folder."""
    executable = shutil.which("tesseract")
    if executable:
        return executable

    windows_executable = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if windows_executable.is_file():
        return str(windows_executable)

    raise FileNotFoundError(
        "Tesseract OCR was not found. Streamlit Cloud installs it from "
        "packages.txt; on Windows, install Tesseract or add it to PATH."
    )


def run_tesseract(image: np.ndarray, executable: str, *config: str) -> str:
    """Run local Tesseract on an OpenCV image and return the recognized text."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image_path = Path(temp_file.name)

    try:
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError("Could not prepare the image for OCR.")
        result = subprocess.run(
            [executable, str(image_path), "stdout", *config],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        return result.stdout
    finally:
        image_path.unlink(missing_ok=True)


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded/captured bytes as a correctly oriented OpenCV BGR image."""
    with Image.open(BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        rgb_image = np.asarray(image)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def prepare_ocr_image(image: np.ndarray) -> np.ndarray:
    """Resize the page to a useful OCR resolution without huge Tesseract inputs."""
    height, width = image.shape[:2]
    scale = min(MAX_OCR_UPSCALE, MAX_OCR_DIMENSION / max(height, width))
    if scale == 1:
        return image
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
    )


def extract_page_words(image: np.ndarray, executable: str) -> list[dict[str, str]]:
    """Read page text and bounding boxes from one Tesseract TSV pass."""
    prepared = prepare_ocr_image(image)
    grayscale = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
    tsv = run_tesseract(grayscale, executable, "--psm", "6", "tsv")
    return [
        row
        for row in csv.DictReader(tsv.splitlines(), delimiter="\t")
        if row.get("text", "").strip()
    ]


def normalize_ocr_digits(value: str) -> str:
    """Correct common letter-shaped OCR substitutions in numeric identifiers."""
    substitutions = {"O": "0", "Q": "0", "I": "1", "L": "1", "|": "1"}
    return value.upper().translate(str.maketrans(substitutions))


def extract_admission_number(
    image: np.ndarray,
    executable: str,
    page_words: list[dict[str, str]] | None = None,
) -> str | None:
    """OCR the full page and locate an ADM-### identifier, correcting common OCR confusions."""
    if page_words is None:
        page_words = extract_page_words(image, executable)

    text = " ".join(word["text"] for word in page_words)
    matches = re.finditer(
        r"\bADM[\s:.-]*([0-9OQIL|]{3})\b", text, re.IGNORECASE
    )
    for match in matches:
        digits = normalize_ocr_digits(match.group(1))
        if digits.isdigit():
            return f"ADM-{digits}"

    # Also handle OCR output that splits ADM from the number or mistakes M for H.
    compact_text = re.sub(r"[^A-Z0-9]", "", text.upper())
    match = re.search(r"AD[MHN]([0-9OQIL|]{3})", compact_text)
    if match:
        digits = normalize_ocr_digits(match.group(1))
        if digits.isdigit():
            return f"ADM-{digits}"
    return None


def find_score_label(
    page_words: list[dict[str, str]],
) -> tuple[str, int, int, int, int, int] | None:
    """Locate a FINAL SCORE label or a GRAND TOTAL label in page OCR results."""
    lines: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for word in page_words:
        key = (word["block_num"], word["par_num"], word["line_num"])
        lines.setdefault(key, []).append(word)

    for words in lines.values():
        line_text = [word["text"].strip(".,:;()").upper() for word in words]
        is_final_score = "FINAL" in line_text and "SCORE" in line_text
        is_grand_total = "GRAND" in line_text and "TOTAL" in line_text
        if is_final_score or is_grand_total:
            label_words = [
                word
                for word in words
                if word["text"].strip(".,:;()").upper()
                in ({"FINAL", "SCORE"} if is_final_score else {"GRAND", "TOTAL"})
            ]
            top = min(int(word["top"]) for word in label_words)
            bottom = max(
                int(word["top"]) + int(word["height"]) for word in label_words
            )
            line_height = max(int(word["height"]) for word in label_words)
            left = min(int(word["left"]) for word in words)
            right = max(int(word["left"]) + int(word["width"]) for word in words)
            kind = "final" if is_final_score else "total"
            return kind, top, bottom, line_height, left, right
    return None


def score_from_red_line(
    red_mask: np.ndarray,
    grayscale: np.ndarray,
    executable: str,
) -> int | None:
    """OCR one horizontal line of red ink and return a plausible score."""
    x, y, width, height = cv2.boundingRect(red_mask)
    selected_mask = red_mask[y : y + height, x : x + width]
    selected_gray = grayscale[y : y + height, x : x + width]
    ocr_image = np.full_like(selected_gray, 255)
    ocr_image[selected_mask > 0] = selected_gray[selected_mask > 0]
    ocr_image = cv2.copyMakeBorder(
        ocr_image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )
    if height < 100:
        scale = min(3.0, 100 / height)
        ocr_image = cv2.resize(
            ocr_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    score_text = run_tesseract(
        ocr_image,
        executable,
        "--psm",
        "8",
        "-c",
        "tessedit_char_whitelist=0123456789/",
    )
    fraction = re.search(r"(?<!\d)(\d{1,3})\s*/\s*\d{1,3}(?!\d)", score_text)
    candidates = (
        [fraction.group(1)]
        if fraction
        else re.findall(r"(?<!\d)\d{1,3}(?!\d)", score_text)
    )
    for candidate in candidates:
        score = int(candidate)
        if 0 <= score <= 100:
            return score
    return None


def first_red_line(
    red_mask: np.ndarray, minimum_gap: int
) -> tuple[int, int] | None:
    """Find the first horizontal band of red pixels, merging tiny row gaps."""
    active_rows = np.flatnonzero(np.any(red_mask > 0, axis=1))
    if active_rows.size == 0:
        return None

    start = previous = int(active_rows[0])
    for row in active_rows[1:]:
        current = int(row)
        if current - previous > minimum_gap + 1:
            return start, previous + 1
        previous = current
    return start, previous + 1


def extract_score(
    image: np.ndarray,
    executable: str,
    page_words: list[dict[str, str]] | None = None,
) -> int | None:
    """OCR the red score, preferring the area identified by its printed label."""
    prepared = prepare_ocr_image(image)
    red_mask = extract_red_mask(prepared)
    if cv2.countNonZero(red_mask) == 0:
        return None

    score_label = find_score_label(page_words or []) if page_words is not None else None
    if score_label is not None:
        label_kind, label_top, label_bottom, label_height, label_left, label_right = score_label
        if label_kind == "final":
            y0 = min(red_mask.shape[0], label_bottom + max(2, label_height // 5))
            y1 = min(
                red_mask.shape[0],
                y0 + max(8 * label_height, int(red_mask.shape[0] * 0.35)),
            )
            label_center = (label_left + label_right) // 2
            half_width = max(
                int(red_mask.shape[1] * 0.23), (label_right - label_left) // 2
            )
            x0 = max(0, label_center - half_width)
            x1 = min(red_mask.shape[1], label_center + half_width)
            roi_mask = red_mask[y0:y1, x0:x1]
            roi_gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
            line = first_red_line(roi_mask, max(3, label_height // 4))
            if line is None:
                return None
            line_top, line_bottom = line
            band_mask = roi_mask[line_top:line_bottom, :]
            band_gray = roi_gray[line_top:line_bottom, :]
            return score_from_red_line(band_mask, band_gray, executable)
        else:
            y0 = max(0, label_top - label_height)
            y1 = min(red_mask.shape[0], label_bottom + label_height)
            red_mask = red_mask[y0:y1, :]
            grayscale = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)[y0:y1, :]
    else:
        grayscale = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)

    if score_label is not None:
        return score_from_red_line(red_mask, grayscale, executable)

    # Without a printed score label there is no reliable way to distinguish a
    # final mark from other red annotations; OCR the whole red region as a fallback.
    x, y, width, height = cv2.boundingRect(red_mask)
    selected_mask = red_mask[y : y + height, x : x + width]
    selected_gray = grayscale[y : y + height, x : x + width]
    ocr_image = np.full_like(selected_gray, 255)
    ocr_image[selected_mask > 0] = selected_gray[selected_mask > 0]
    ocr_image = cv2.copyMakeBorder(
        ocr_image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )
    score_text = run_tesseract(
        ocr_image,
        executable,
        "--psm",
        "11",
        "-c",
        "tessedit_char_whitelist=0123456789/",
    )
    candidates = re.findall(r"(?<!\d)\d{1,3}(?!\d)", score_text)
    for candidate in candidates:
        score = int(candidate)
        if 0 <= score <= 100:
            return score
    return None


def extract_red_mask(image: np.ndarray) -> np.ndarray:
    """Isolate red pen pixels using both red hue ranges in HSV."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 50, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)




def grade_for_score(score: int) -> str:
    """Map a valid mark to the requested letter grade."""
    if not 0 <= score <= 100:
        raise ValueError("Score must be between 0 and 100.")
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def scan_exam(
    image_bytes: bytes, students: dict[str, dict[str, str]]
) -> dict[str, str | int | None]:
    """Decode and scan one exam image, returning student and mark details."""
    image = decode_image(image_bytes)
    executable = find_tesseract()
    page_words = extract_page_words(image, executable)
    admission_no = extract_admission_number(image, executable, page_words)
    score = extract_score(image, executable, page_words)
    student = students.get(admission_no or "", {})
    return {
        "Admission Number": admission_no or "Not detected",
        "Student Name": student.get("Student_Name") or "Unknown Student",
        "Class": student.get("Class", ""),
        "Score": score,
        "Grade": grade_for_score(score) if score is not None else "Not detected",
    }


def main() -> None:
    st.set_page_config(page_title="School Exam Scanner MVP", page_icon="📚")
    st.title("📚 School Exam Scanner MVP")
    st.write(
        "Upload an exam cover page or take a photo. The app reads the admission "
        "number and isolates red-ink marks using local OCR."
    )

    try:
        students = load_students()
    except (OSError, csv.Error, ValueError) as error:
        st.error(f"Could not load {STUDENTS_CSV.name}: {error}")
        st.stop()

    input_method = st.radio(
        "Choose how to provide an exam cover",
        ("Upload an image", "Use camera"),
        horizontal=True,
    )
    if input_method == "Upload an image":
        image_file = st.file_uploader(
            "Choose an exam cover image",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        )
    else:
        image_file = st.camera_input("Take a photo of the exam cover")

    if image_file is None:
        st.info("Provide an image to scan.")
        return

    image_bytes = image_file.getvalue()
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        st.error("Image is too large. Please use an image smaller than 20 MB.")
        return

    st.image(image_bytes, caption="Exam cover to scan", use_container_width=True)
    if not st.button("Scan exam cover", type="primary"):
        return

    try:
        result = scan_exam(image_bytes, students)
    except FileNotFoundError as error:
        st.error(str(error))
        return
    except subprocess.TimeoutExpired:
        st.error("Tesseract OCR took too long. Try a smaller or clearer image.")
        return
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() if error.stderr else "No Tesseract error details."
        st.error(f"Tesseract OCR failed: {details}")
        return
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Could not process this image: {error}")
        return

    st.subheader("Scan result")
    admission_col, name_col, score_col, grade_col = st.columns(4)
    admission_col.metric("Admission Number", result["Admission Number"])
    name_col.metric("Student Name", result["Student Name"])
    score_col.metric(
        "Extracted Score",
        result["Score"] if result["Score"] is not None else "Not detected",
    )
    grade_col.metric("Assigned Grade", result["Grade"])
    if result["Admission Number"] == "Not detected":
        st.warning("Admission number was not detected. Check the image and try again.")
    elif result["Student Name"] == "Unknown Student":
        st.warning("Admission number was not found in students_db.csv.")
    if result["Score"] is None:
        st.warning("No red-ink score was detected. Review the image and try again.")
    elif result["Class"]:
        st.caption(f"Class: {result['Class']}")


if __name__ == "__main__":
    main()
