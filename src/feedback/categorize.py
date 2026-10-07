"""
Feedback engine: explains the relation between a flagged word and a suggestion.

Each (word, suggestion) pair is assigned one of four categories:
    accent        the two forms differ only in their accents
    graphematic   the difference matches a confusion set of the graphematic repertoire
    typing        a single typing slip (swapped, wrong, extra or missing letter)
    contextual    the word is itself a valid word, or shows no recognisable slip
"""

import unicodedata

from src.config import CONTEXT_FIT_GOOD, CONTEXT_FIT_STRONG, MAX_SUGGESTIONS_SHOWN
from src.feedback.messages import EXPLANATIONS
from src.rules import GRAPHEMATIC_RULES, KEYBOARD_NEIGHBOURS, grapheme_family

# Longer graphemes first, so that 'tz' is recognised before 'z'
_RULES_BY_LENGTH = sorted(GRAPHEMATIC_RULES.items(), key=lambda item: len(item[0]), reverse=True)


def strip_accents(word: str) -> str:
    nfkd = unicodedata.normalize('NFKD', word)
    return "".join(char for char in nfkd if not unicodedata.combining(char))


def differing_parts(word: str, suggestion: str) -> tuple:
    """ The parts of the two forms left once their common beginning and ending are removed. """
    max_common = min(len(word), len(suggestion))
    prefix = 0
    while prefix < max_common and word[prefix] == suggestion[prefix]:
        prefix += 1
    suffix = 0
    while suffix < max_common - prefix and word[-1 - suffix] == suggestion[-1 - suffix]:
        suffix += 1
    return word[prefix:len(word) - suffix], suggestion[prefix:len(suggestion) - suffix]


def find_accent_difference(word, suggestion):
    """ Returns (pattern, wrong, right) when the two forms differ only in their accents. """
    if word == suggestion or strip_accents(word) != strip_accents(suggestion):
        return None

    wrong, right = differing_parts(word, suggestion)
    if strip_accents(right) == right:
        return "accent_extra", wrong, right
    if strip_accents(wrong) == wrong:
        return "accent_missing", wrong, right
    return "accent_changed", wrong, right


def find_graphematic_confusion(word, suggestion):
    """ Returns (family, wrong, right) when a confusion set turns the suggestion into the word. """
    # The correct grapheme of the suggestion was replaced by one of its usual confusions...
    for correct, confusions in _RULES_BY_LENGTH:
        for confusion in confusions:
            start = suggestion.find(correct)
            while start != -1:
                if suggestion[:start] + confusion + suggestion[start + len(correct):] == word:
                    return grapheme_family(correct, confusion), confusion, correct
                start = suggestion.find(correct, start + 1)

    # ...or the other way round (e.g. 'cc' written where 'c' is expected)
    for correct, confusions in _RULES_BY_LENGTH:
        for confusion in confusions:
            start = word.find(correct)
            while start != -1:
                if word[:start] + confusion + word[start + len(correct):] == suggestion:
                    return grapheme_family(correct, confusion), correct, confusion
                start = word.find(correct, start + 1)

    return None


def find_typing_slip(word, suggestion):
    """ Returns (pattern, wrong, right) when the two forms are one typing slip apart. """
    wrong, right = differing_parts(word, suggestion)

    if len(wrong) == 2 and len(right) == 2 and wrong == right[::-1]:
        return "swapped_letters", wrong, right
    if len(wrong) == 1 and len(right) == 1:
        if wrong in KEYBOARD_NEIGHBOURS.get(right, []):
            return "neighbouring_key", wrong, right
        return "wrong_letter", wrong, right
    if len(wrong) == 1 and right == "":
        return "extra_letter", wrong, right
    if wrong == "" and len(right) == 1:
        return "missing_letter", wrong, right
    return None


def categorize(word: str, suggestion: str, vocabulary=frozenset()) -> dict:
    """ Category, pattern and explanation for a flagged word and one of its suggestions. """
    written, proposed = word.lower(), suggestion.lower()

    # Form of the slip, from the most to the least specific
    form = None
    for category, finder in (("accent", find_accent_difference),
                             ("graphematic", find_graphematic_confusion),
                             ("typing", find_typing_slip)):
        found = finder(written, proposed)
        if found:
            form = (category, *found)
            break

    # A valid word can only have been flagged because of its context
    if written in vocabulary:
        template = "real_word_with_hint" if form and form[0] == "graphematic" else "real_word"
        wrong, right = (form[2], form[3]) if form else ("", "")
        return {
            "category": "contextual",
            "pattern": "real_word",
            "explanation": EXPLANATIONS[template].format(word=word, suggestion=suggestion, wrong=wrong, right=right)
        }

    if form is None:
        return {
            "category": "contextual",
            "pattern": "closest_word",
            "explanation": EXPLANATIONS["closest_word"].format(word=word, suggestion=suggestion)
        }

    category, pattern, wrong, right = form
    return {
        "category": category,
        "pattern": pattern,
        "explanation": EXPLANATIONS[pattern].format(word=word, suggestion=suggestion, wrong=wrong, right=right)
    }


def context_fit(gain: float) -> str:
    """ Turns the language model gain of a suggestion over the written word into a level. """
    if gain >= CONTEXT_FIT_STRONG:
        return "strong"
    if gain >= CONTEXT_FIT_GOOD:
        return "good"
    return "weak"


def build_issues(analysis, vocabulary, offset=0) -> list:
    """
    Turns the analysis of the spellchecker into the list of issues shown to the writer:
    flagged words with their explained suggestions, and words the detector considers
    doubtful but for which no suggestion is available.
    """
    issues = []
    for chunk in analysis:
        if chunk["status"] == "ok":
            continue

        issue = {
            "start": chunk["start"] + offset,
            "end": chunk["end"] + offset,
            "word": chunk["word"],
            "status": chunk["status"],
            "suggestions": []
        }

        if chunk["status"] == "suggestion":
            for candidate, gain in list(zip(chunk["candidates"], chunk["context_gains"]))[:MAX_SUGGESTIONS_SHOWN]:
                issue["suggestions"].append({
                    "word": candidate,
                    "fit": context_fit(gain),
                    **categorize(chunk["word"], candidate, vocabulary)
                })
            # The issue takes the category of its first suggestion
            best = issue["suggestions"][0]
            issue.update(category=best["category"], pattern=best["pattern"], explanation=best["explanation"])
        else:
            issue.update(category="unverified", pattern="no_suggestion", explanation=EXPLANATIONS["no_suggestion"])

        issues.append(issue)
    return issues
