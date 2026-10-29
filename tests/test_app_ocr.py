import sys
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))

import app


class ScoreOcrTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = np.zeros((80, 140), dtype=np.uint8)
        cv2.rectangle(self.mask, (45, 15), (90, 65), 255, thickness=-1)
        self.grayscale = np.full(self.mask.shape, 255, dtype=np.uint8)

    def scan_with_ocr(self, readings: list[str]) -> int | None:
        with patch("app.run_tesseract", side_effect=readings):
            return app.score_from_red_line(self.mask, self.grayscale, "tesseract")

    def test_recovers_full_85_when_primary_ocr_drops_first_digit(self) -> None:
        self.assertEqual(self.scan_with_ocr(["5", "85", "85"]), 85)

    def test_recovers_full_72_when_primary_ocr_drops_first_digit(self) -> None:
        self.assertEqual(self.scan_with_ocr(["2", "72", "72"]), 72)

    def test_keeps_a_genuine_single_digit_five(self) -> None:
        self.assertEqual(self.scan_with_ocr(["5", "5", "5"]), 5)

    def test_keeps_zero_as_a_valid_mark(self) -> None:
        self.assertEqual(self.scan_with_ocr(["0", "0", "0"]), 0)

    def test_accepts_complete_scores_from_zero_through_one_hundred(self) -> None:
        for score in (0, 1, 5, 8, 9, 10, 11, 21, 45, 50, 75, 80, 99, 100):
            with self.subTest(score=score):
                text = str(score)
                self.assertEqual(self.scan_with_ocr([text, text, text]), score)

    def test_returns_no_score_when_ocr_attempts_conflict(self) -> None:
        self.assertIsNone(self.scan_with_ocr(["5", "45", "75"]))

    def test_does_not_accept_a_single_digit_against_one_double_digit_reading(self) -> None:
        self.assertIsNone(self.scan_with_ocr(["5", "45", "5"]))

    def test_ocr_uses_upscaled_crop_and_digit_only_line_mode(self) -> None:
        with patch("app.run_tesseract", side_effect=["5", "5", "5"]) as run_ocr:
            app.score_from_red_line(self.mask, self.grayscale, "tesseract")

        first_call = run_ocr.call_args_list[0]
        self.assertEqual(
            first_call.args[2:],
            ("--psm", "7", "-c", "tessedit_char_whitelist=0123456789"),
        )
        self.assertEqual(first_call.args[0].shape, (198, 240))

    def test_rejects_out_of_range_numbers_without_truncating_them(self) -> None:
        self.assertEqual(app.valid_score_values(["100", "101", "178"]), [100])
        self.assertEqual(app.valid_score_values(["101", "178"]), [])

    def test_score_candidates_preserve_the_complete_number(self) -> None:
        self.assertEqual(app.score_candidates("85"), ["85"])
        self.assertEqual(app.score_candidates("21"), ["21"])
        self.assertEqual(app.score_candidates("100"), ["100"])

    def test_student_database_maps_all_four_admissions(self) -> None:
        students = app.load_students()
        self.assertEqual(
            {admission: student["Student_Name"] for admission, student in students.items()},
            {
                "ADM-101": "John Doe",
                "ADM-102": "Jane Smith",
                "ADM-103": "Alex Otieno",
                "ADM-104": "Mary Wanjiku",
            },
        )

    def test_grade_boundaries(self) -> None:
        self.assertEqual(app.grade_for_score(0), "D")
        self.assertEqual(app.grade_for_score(50), "C")
        self.assertEqual(app.grade_for_score(65), "B")
        self.assertEqual(app.grade_for_score(80), "A")
        self.assertEqual(app.grade_for_score(100), "A")


if __name__ == "__main__":
    unittest.main()
