"""Legacy profile-parser contract, tested with synthetic source text."""

import unittest

from utils import parse_aim_resumes


class ParseAimResumeTests(unittest.TestCase):
    def test_expected_section_order(self):
        source = (
            "SUMMARY\nSynthetic summary\n"
            "SKILLS AND TECHNOLOGIES\nPython\n"
            "PROFESSIONAL EXPERIENCE\nExample Co\n"
            "EDUCATION\nExample University\n"
            "CERTIFICATIONS\nExample certification\n"
            "AWARDS\nExample award"
        )
        self.assertEqual(parse_aim_resumes(source), (
            "Synthetic summary", "Python", "Example Co", "Example University",
            "Example certification", "Example award",
        ))

    def test_missing_sections_are_not_fabricated(self):
        self.assertEqual(parse_aim_resumes("SUMMARY\nSynthetic summary"), (
            "Synthetic summary", None, None, None, None, None,
        ))


if __name__ == "__main__":
    unittest.main()
