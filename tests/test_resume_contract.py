"""Synthetic, offline acceptance tests for the extraction/renderer boundary."""

from copy import deepcopy
import unittest

from resume_contract import (
    MAX_BULLETS_PER_JOB, MAX_EDUCATION_ITEMS, MAX_JOBS, MAX_SKILLS,
    MAX_TOTAL_TEXT_CHARS, ResumeValidationError, SECTION_KEYS,
    to_section_results, validate_resume,
)


def empty_resume():
    return {
        "name": "", "summary": "", "skills_and_tech": [],
        "professional_experience": [], "education": [],
    }


def job(**overrides):
    result = {
        "employer": "", "location": "", "title": "", "start_date": "",
        "end_date": "", "bullets": [],
    }
    result.update(overrides)
    return result


class ResumeContractTests(unittest.TestCase):
    def test_full_resume_preserves_exact_legacy_shape(self):
        data = {
            "name": " Avery Example ", "summary": "Builds reliable tools.",
            "skills_and_tech": ["Python", "SQL"],
            "professional_experience": [job(
                employer="Example Works", location="Remote", title="Engineer",
                start_date="2021", end_date="2024", bullets=["Built an internal tool."]
            )],
            "education": ["Example College, B.S., 2020"],
        }
        self.assertEqual(to_section_results(data), {
            "name": "Avery Example", "summary": "Builds reliable tools.",
            "skills_and_tech": "Python\nSQL",
            "professional_experience": "Example Works, Remote\t2021 - 2024\nEngineer\nBuilt an internal tool.",
            "education": "Example College, B.S., 2020",
        })

    def test_all_missing_facts_stay_empty(self):
        data = empty_resume()
        data["professional_experience"] = [job()]
        self.assertEqual(to_section_results(data), dict.fromkeys(SECTION_KEYS, ""))

    def test_missing_job_headers_keep_two_slots(self):
        data = empty_resume()
        data["professional_experience"] = [job(bullets=["Maintained a tool."])]
        self.assertEqual(to_section_results(data)["professional_experience"], "\t\n\t\nMaintained a tool.")

    def test_single_dates_do_not_invent_range_endpoints(self):
        for dates, expected in [
            ({"start_date": "2020"}, "2020"),
            ({"end_date": "2024"}, "2024"),
            ({"start_date": "2020", "end_date": "Present"}, "2020 - Present"),
        ]:
            with self.subTest(dates=dates):
                data = empty_resume()
                data["professional_experience"] = [job(**dates)]
                self.assertEqual(to_section_results(data)["professional_experience"], expected + "\n\t")

    def test_multiple_jobs_keep_unambiguous_boundaries(self):
        data = empty_resume()
        data["professional_experience"] = [
            job(employer="Example One", bullets=["Built a tool."]),
            job(title="Analyst", bullets=["Reviewed reports."]),
        ]
        blocks = to_section_results(data)["professional_experience"].split("\n\n")
        self.assertEqual(blocks, ["Example One\n\t\nBuilt a tool.", "\t\nAnalyst\nReviewed reports."])

    def test_word_bullets_do_not_destroy_technical_punctuation(self):
        data = empty_resume()
        data["skills_and_tech"] = ["• .NET", "- Node.js", "* C++", "Python 3.12", "-3dB measurements"]
        data["professional_experience"] = [job(bullets=["• Used Node.js."])]
        output = to_section_results(data)
        self.assertEqual(output["skills_and_tech"], ".NET\nNode.js\nC++\nPython 3.12\n-3dB measurements")
        self.assertEqual(output["professional_experience"], "\t\n\t\nUsed Node.js.")

    def test_blank_list_items_and_whitespace_are_normalized(self):
        data = empty_resume()
        data["skills_and_tech"] = [" ", " • ", " SQL "]
        data["education"] = ["", " Example College "]
        output = to_section_results(data)
        self.assertEqual(output["skills_and_tech"], "SQL")
        self.assertEqual(output["education"], "Example College")

    def test_summary_paragraphs_normalize_line_endings(self):
        data = empty_resume()
        data["summary"] = " First paragraph.\r\n\r\nSecond paragraph. "
        self.assertEqual(to_section_results(data)["summary"], "First paragraph.\n\nSecond paragraph.")

    def test_missing_extra_and_nonobject_top_level_are_rejected(self):
        missing = empty_resume()
        del missing["name"]
        extra = empty_resume()
        extra["awards"] = []
        for data in [missing, extra, [], None, "not an object"]:
            with self.subTest(data_type=type(data).__name__):
                with self.assertRaises(ResumeValidationError):
                    validate_resume(data)

    def test_wrong_scalar_and_array_types_are_rejected(self):
        cases = [("name", None), ("summary", False), ("skills_and_tech", "SQL"),
                 ("skills_and_tech", [1]), ("education", [None]),
                 ("professional_experience", {}), ("professional_experience", [False])]
        for key, value in cases:
            with self.subTest(key=key, value_type=type(value).__name__):
                data = empty_resume()
                data[key] = value
                with self.assertRaises(ResumeValidationError):
                    validate_resume(data)

    def test_job_schema_is_exact(self):
        missing = job()
        del missing["end_date"]
        for value in [missing, job(extra="unexpected"), job(start_date=2020), job(bullets=[True])]:
            data = empty_resume()
            data["professional_experience"] = [value]
            with self.assertRaises(ResumeValidationError):
                validate_resume(data)

    def test_metadata_cannot_inject_document_lines(self):
        for separator in ["\n", "\r", "\r\n", "\t", "\v", "\u2028", "\u2029"]:
            for field in ["employer", "location", "title", "start_date", "end_date"]:
                with self.subTest(separator=repr(separator), field=field):
                    data = empty_resume()
                    data["professional_experience"] = [job(**{field: "before" + separator + "after"})]
                    with self.assertRaises(ResumeValidationError):
                        validate_resume(data)
        for key, value in [("name", "A\nB"), ("skills_and_tech", ["A\nB"]),
                           ("education", ["A\nB"])]:
            data = empty_resume()
            data[key] = value
            with self.assertRaises(ResumeValidationError):
                validate_resume(data)

    def test_invalid_text_never_appears_in_error_messages(self):
        sentinel = "SYNTHETIC_PRIVATE_SENTINEL"
        for invalid in [sentinel + "\x00", sentinel + "\n"]:
            data = empty_resume()
            data["name"] = invalid
            with self.assertRaises(ResumeValidationError) as caught:
                validate_resume(data)
            self.assertNotIn(sentinel, str(caught.exception))
        data = empty_resume()
        data[sentinel] = "unexpected"
        with self.assertRaises(ResumeValidationError) as caught:
            validate_resume(data)
        self.assertNotIn(sentinel, str(caught.exception))

    def test_xml_forbidden_characters_are_rejected_before_rendering(self):
        for invalid in ("\ud800", "\udfff", "\ufffe", "\uffff", "\x00"):
            for field in ("name", "summary", "skills_and_tech", "professional_experience"):
                with self.subTest(codepoint=hex(ord(invalid)), field=field):
                    data = empty_resume()
                    if field == "skills_and_tech":
                        data[field] = ["SYNTHETIC_SENTINEL" + invalid]
                    elif field == "professional_experience":
                        data[field] = [job(bullets=["SYNTHETIC_SENTINEL" + invalid])]
                    else:
                        data[field] = "SYNTHETIC_SENTINEL" + invalid
                    with self.assertRaises(ResumeValidationError) as caught:
                        to_section_results(data)
                    self.assertNotIn("SYNTHETIC_SENTINEL", str(caught.exception))

    def test_valid_supplementary_unicode_and_summary_newlines_are_preserved(self):
        data = empty_resume()
        data["name"] = "Avery \U0001f680"
        data["summary"] = "Built a tool \U0001f600\r\n\r\nSecond paragraph."
        data["skills_and_tech"] = ["Unicode \U00010000"]
        output = to_section_results(data)
        self.assertEqual(output["name"], "Avery \U0001f680")
        self.assertEqual(output["summary"], "Built a tool \U0001f600\n\nSecond paragraph.")
        self.assertEqual(output["skills_and_tech"], "Unicode \U00010000")

    def test_array_limits_allow_boundary_and_reject_excess(self):
        for key, maximum, item in [
            ("skills_and_tech", MAX_SKILLS, "SQL"),
            ("education", MAX_EDUCATION_ITEMS, "Example College"),
            ("professional_experience", MAX_JOBS, job()),
        ]:
            with self.subTest(key=key):
                data = empty_resume()
                data[key] = [deepcopy(item) for _ in range(maximum)]
                validate_resume(data)
                data[key].append(deepcopy(item))
                with self.assertRaises(ResumeValidationError):
                    validate_resume(data)
        data = empty_resume()
        data["professional_experience"] = [job(bullets=["Built a tool."] * MAX_BULLETS_PER_JOB)]
        validate_resume(data)
        data["professional_experience"][0]["bullets"].append("One too many.")
        with self.assertRaises(ResumeValidationError):
            validate_resume(data)

    def test_field_and_total_text_limits(self):
        data = empty_resume()
        data["summary"] = "x" * MAX_TOTAL_TEXT_CHARS
        validate_resume(data)
        data["name"] = "A"
        with self.assertRaises(ResumeValidationError):
            validate_resume(data)
        for field, value in [("name", "x" * 501), ("skills_and_tech", ["x" * 5001])]:
            data = empty_resume()
            data[field] = value
            with self.assertRaises(ResumeValidationError):
                validate_resume(data)
        data = empty_resume()
        data["professional_experience"] = [job(start_date="x" * 101)]
        with self.assertRaises(ResumeValidationError):
            validate_resume(data)

    def test_validation_and_adaptation_do_not_mutate_input(self):
        data = empty_resume()
        data["professional_experience"] = [job(title=" Engineer ", bullets=[" Built a tool. "])]
        before = deepcopy(data)
        normalized = validate_resume(data)
        to_section_results(data)
        self.assertEqual(data, before)
        normalized["professional_experience"][0]["bullets"].append("Independent change.")
        self.assertEqual(data, before)


if __name__ == "__main__":
    unittest.main()
