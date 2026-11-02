"""Scan exam cover images and export student marks to an Excel workbook."""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from pathlib import Path
import shutil
from typing import Optional

import cv2
import numpy as np
from openpyxl import Workbook


BASE_DIR = Path(__file__).resolve().parent
EXAMS_DIR = BASE_DIR / "sample_exams"
STUDENTS_CSV = BASE_DIR / "students_db.csv"
OUTPUT_XLSX = BASE_DIR / "final_marks.xlsx"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUTPUT_COLUMNS = [
    "Exam_File",
    "Admission_No",
    "Student_Name",
    "Class",
    "Score",
    "Grade",
    "Status",
]


def create_students_database_if_missing() -> None:
    """Create the example student database without overwriting an existing one."""
    if STUDENTS_CSV.exists():
        return

    sample_students = [
        {"Admission_No": "ADM-101", "Student_Name": "John Doe", "Class": "Form 3 East"},
        {"Admission_No": "ADM-102", "Student_Name": "Jane Smith", "Class": "Form 3 West"},
        {"Admission_No": "ADM-103", "Student_Name": "Alex Otieno", "Class": "Form 3 North"},
        {"Admission_No": "ADM-104", "Student_Name": "Mary Wanjiku", "Class": "Form 3 South"},
    ]
    with STUDENTS_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["Admission_No", "Student_Name", "Class"]
        )
        writer.writeheader()
        writer.writerows(sample_students)
    print(f"Created sample student database: {STUDENTS_CSV.name}")


def configure_tesseract() -> Path:
    """Find the Tesseract executable, including its standard Windows install path."""
    executable = shutil.which("tesseract")
    if executable is None:
        standard_windows_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if standard_windows_path.is_file():
            executable = str(standard_windows_path)

    if executable is None:
        raise FileNotFoundError(
            "Tesseract OCR was not found. Add its install directory to PATH or "
            "install it from UB-Mannheim's Tesseract-OCR package."
        )
    return Path(executable)


def run_tesseract(image: np.ndarray, executable: Path, *config: str) -> str:
    """Run the local Tesseract executable on an image and return recognized text."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
        image_path = Path(temp_image.name)

    try:
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Could not write temporary OCR image: {image_path}")
        result = subprocess.run(
            [str(executable), str(image_path), "stdout", *config],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout
    finally:
        image_path.unlink(missing_ok=True)


def normalize_admission_number(value: str) -> Optional[str]:
    """Return a canonical ADM-### value, or None when the value is malformed."""
    compact = re.sub(r"[\s-]", "", value.upper())
    match = re.fullmatch(r"ADM(\d{3})", compact)
    return f"ADM-{match.group(1)}" if match else None


def load_students() -> dict[str, dict[str, str]]:
    """Load student records keyed by their canonical admission number."""
    required_columns = {"Admission_No", "Student_Name", "Class"}
    lookup: dict[str, dict[str, str]] = {}
    with STUDENTS_CSV.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"{STUDENTS_CSV.name} is missing required column(s): {missing}"
            )
        for row in reader:
            admission_no = normalize_admission_number(row["Admission_No"] or "")
            if admission_no:
                lookup[admission_no] = {
                    "Student_Name": (row["Student_Name"] or "").strip(),
                    "Class": (row["Class"] or "").strip(),
                }
    return lookup


def extract_red_marks(image: np.ndarray) -> np.ndarray:
    """Return a binary mask of pixels within the two red HSV ranges."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red_mask = cv2.inRange(
        hsv_image, np.array([0, 50, 50]), np.array([10, 255, 255])
    )
    upper_red_mask = cv2.inRange(
        hsv_image, np.array([170, 50, 50]), np.array([180, 255, 255])
    )
    red_mask = cv2.bitwise_or(lower_red_mask, upper_red_mask)
    return red_mask


def extract_admission_number(image: np.ndarray, executable: Path) -> Optional[str]:
    """Find an ADM-### or ADM### identifier using OCR on the full page."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_text = run_tesseract(grayscale, executable, "--psm", "6")
    combined_text = re.sub(r"[\s-]", "", detected_text.upper())
    match = re.search(r"ADM(\d{3})", combined_text)
    return f"ADM-{match.group(1)}" if match else None


def extract_score_from_red_marks(
    image: np.ndarray, executable: Path
) -> Optional[int]:
    """Read a score from the red-only image; ignore non-score numeric tokens."""
    red_only = extract_red_marks(image)
    if not np.any(red_only):
        return None

    x, y, width, height = cv2.boundingRect(red_only)
    cropped_mask = red_only[y : y + height, x : x + width]
    ocr_image = cv2.bitwise_not(cropped_mask)
    ocr_image = cv2.resize(
        ocr_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )
    detected_text = run_tesseract(
        ocr_image,
        executable,
        "--psm",
        "6",
        "-c",
        "tessedit_char_whitelist=0123456789/",
    )
    for token in re.findall(r"\d{1,3}", detected_text):
        score = int(token)
        if 0 <= score <= 100:
            return score
    return None


def grade_for_score(score: int) -> str:
    """Convert a validated score to the requested letter grade."""
    if not 0 <= score <= 100:
        raise ValueError(f"Score must be between 0 and 100; received {score}.")
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def process_exam(
    image_path: Path,
    students: dict[str, dict[str, str]],
    executable: Path,
) -> dict[str, object]:
    """Process one cover image and return its export-ready record."""
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "Exam_File": image_path.name,
            "Admission_No": "",
            "Student_Name": "Unknown Student",
            "Class": "",
            "Score": None,
            "Grade": "",
            "Status": "Image could not be read",
        }

    admission_no = extract_admission_number(image, executable)
    student = students.get(admission_no, {}) if admission_no else {}
    score = extract_score_from_red_marks(image, executable)

    status_messages = []
    if not admission_no:
        status_messages.append("Admission number not detected")
    elif not student:
        status_messages.append("Admission number not found in database")
    if score is None:
        status_messages.append("Red-ink score not detected")

    return {
        "Exam_File": image_path.name,
        "Admission_No": admission_no or "",
        "Student_Name": student.get("Student_Name") or "Unknown Student",
        "Class": student.get("Class", ""),
        "Score": score,
        "Grade": grade_for_score(score) if score is not None else "",
        "Status": "; ".join(status_messages) if status_messages else "OK",
    }


def main() -> None:
    create_students_database_if_missing()
    EXAMS_DIR.mkdir(parents=True, exist_ok=True)
    students = load_students()
    executable = configure_tesseract()
    exam_images = sorted(
        path
        for path in EXAMS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not exam_images:
        print(f"No exam images found. Add image files to: {EXAMS_DIR}")
        save_results([], OUTPUT_XLSX)
        print(f"Created empty results workbook: {OUTPUT_XLSX.name}")
        return

    records = [process_exam(path, students, executable) for path in exam_images]

    save_results(records, OUTPUT_XLSX)
    print(f"Processed {len(records)} exam image(s).")
    print(f"Results saved to: {OUTPUT_XLSX}")
    print("Review rows whose Status is not 'OK' before using the marks.")


def save_results(records: list[dict[str, object]], output_path: Path) -> None:
    """Write scan records to an Excel workbook without pandas."""
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Final Marks"
    worksheet.append(OUTPUT_COLUMNS)
    for record in records:
        worksheet.append([record.get(column) for column in OUTPUT_COLUMNS])
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    workbook.save(output_path)


if __name__ == "__main__":
    main()
