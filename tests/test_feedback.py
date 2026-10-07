"""
Tests of the feedback engine (no models needed).

Usage:
    python -m unittest discover tests
"""

import unittest

from src.feedback.categorize import build_issues, categorize, differing_parts


class CategorizeTest(unittest.TestCase):

    def assertFeedback(self, word, suggestion, category, pattern, vocabulary=frozenset()):
        feedback = categorize(word, suggestion, vocabulary)
        self.assertEqual((feedback["category"], feedback["pattern"]), (category, pattern), feedback["explanation"])
        return feedback

    def test_differing_parts(self):
        self.assertEqual(differing_parts("castedu", "casteddu"), ("", "d"))
        self.assertEqual(differing_parts("msesudì", "mesudì"), ("s", ""))
        self.assertEqual(differing_parts("ceh", "che"), ("eh", "he"))

    def test_accents(self):
        self.assertFeedback("mesudi", "mesudì", "accent", "accent_missing")
        self.assertFeedback("còsa", "cosa", "accent", "accent_extra")
        self.assertFeedback("perchè", "perché", "accent", "accent_changed")

    def test_graphematic_confusions(self):
        self.assertFeedback("castedu", "casteddu", "graphematic", "double_consonant")
        self.assertFeedback("fueddu", "fuedu", "graphematic", "double_consonant")
        self.assertFeedback("pitzu", "pizu", "graphematic", "sibilant")
        self.assertFeedback("sceti", "xeti", "graphematic", "italian_conflict")
        self.assertFeedback("melì", "merì", "graphematic", "similar_sound")

    def test_longer_grapheme_wins(self):
        feedback = self.assertFeedback("pizu", "pitzu", "graphematic", "sibilant")
        self.assertIn("“tz”", feedback["explanation"])

    def test_typing_slips(self):
        self.assertFeedback("ceh", "che", "typing", "swapped_letters")
        self.assertFeedback("trwnu", "trenu", "typing", "neighbouring_key")
        self.assertFeedback("trunu", "trenu", "typing", "wrong_letter")
        self.assertFeedback("msesudì", "mesudì", "typing", "extra_letter")
        self.assertFeedback("mesdì", "mesudì", "typing", "missing_letter")

    def test_valid_word_is_contextual(self):
        feedback = self.assertFeedback("ci", "chi", "contextual", "real_word", vocabulary={"ci", "chi"})
        self.assertIn("easily confused", feedback["explanation"])
        self.assertFeedback("cai", "fai", "contextual", "real_word", vocabulary={"cai", "fai"})

    def test_no_recognisable_slip(self):
        self.assertFeedback("xanusai", "lanusei", "contextual", "closest_word")

    def test_case_is_ignored_but_kept_in_the_explanation(self):
        feedback = self.assertFeedback("Mesudi", "Mesudì", "accent", "accent_missing")
        self.assertIn("Mesudì", feedback["explanation"])


class BuildIssuesTest(unittest.TestCase):

    ANALYSIS = [
        {"word": "su", "status": "ok", "candidates": [], "start": 0, "end": 2},
        {"word": " ", "status": "ok", "candidates": [], "start": 2, "end": 3},
        {"word": "castedu", "status": "suggestion", "candidates": ["casteddu", "castedus"],
         "context_gains": [3.1, -0.4], "start": 3, "end": 10},
        {"word": " ", "status": "ok", "candidates": [], "start": 10, "end": 11},
        {"word": "xyzw", "status": "unverified", "candidates": [], "start": 11, "end": 15},
    ]

    def test_issues(self):
        issues = build_issues(self.ANALYSIS, vocabulary=set(), offset=100)
        self.assertEqual(len(issues), 2)

        flagged, doubtful = issues
        self.assertEqual((flagged["start"], flagged["end"]), (103, 110))
        self.assertEqual(flagged["category"], "graphematic")
        self.assertEqual([s["word"] for s in flagged["suggestions"]], ["casteddu", "castedus"])
        self.assertEqual([s["fit"] for s in flagged["suggestions"]], ["strong", "weak"])

        self.assertEqual(doubtful["category"], "unverified")
        self.assertEqual(doubtful["suggestions"], [])


if __name__ == "__main__":
    unittest.main()
