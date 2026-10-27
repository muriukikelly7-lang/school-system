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


def extract_admission_number(image: np.ndarray, executable: str) -> str | None:
    """OCR the full grayscale page and locate an ADM-### student identifier."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    page_text = run_tesseract(grayscale, executable, "--psm", "6")

    for line in page_text.splitlines():
        match = re.search(r"\bADM\s*-?\s*(\d{3})\b", line, re.IGNORECASE)
        if match:
            return f"ADM-{match.group(1)}"

    # Also handle OCR output that separates the prefix and digits across lines.
    compact_text = re.sub(r"[\s-]", "", page_text.upper())
    match = re.search(r"ADM(\d{3})", compact_text)
    return f"ADM-{match.group(1)}" if match else None


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


def extract_score(image: np.ndarray, executable: str) -> int | None:
    """OCR only red-mask pixels and return the first plausible score (0-100)."""
    red_mask = extract_red_mask(image)
    if cv2.countNonZero(red_mask) == 0:
        return None

    x, y, width, height = cv2.boundingRect(red_mask)
    cropped_mask = red_mask[y : y + height, x : x + width]
    # Tesseract expects dark glyphs on a light background.
    ocr_image = cv2.bitwise_not(cropped_mask)
    ocr_image = cv2.copyMakeBorder(
        ocr_image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )
    ocr_image = cv2.resize(
        ocr_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )
    score_text = run_tesseract(
        ocr_image,
        executable,
        "--psm",
        "6",
        "-c",
        "tessedit_char_whitelist=0123456789/",
    )

    fraction = re.search(r"(?<!\d)(\d{1,3})\s*/\s*\d{1,3}(?!\d)", score_text)
    candidates = (
        [fraction.group(1)]
        if fraction
        else re.findall(r"\b\d{1,3}\b", score_text)
    )
    for candidate in candidates:
        score = int(candidate)
        if 0 <= score <= 100:
            return score
    return None


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
    admission_no = extract_admission_number(image, executable)
    score = extract_score(image, executable)
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
