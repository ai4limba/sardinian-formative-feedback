"""
Every text shown to the writer, in one place: feedback categories, explanations and
interface labels. Translating the tool means translating this file.
"""

# ==========================================
# FEEDBACK CATEGORIES
# ==========================================
CATEGORIES = {
    "accent": {
        "label": "Accent",
        "description": "The word differs from the suggestion only in its accent."
    },
    "graphematic": {
        "label": "Graphematic confusion",
        "description": "Two spellings that stand for similar sounds have been mixed up."
    },
    "typing": {
        "label": "Typing error",
        "description": "A letter was swapped, added, left out or mistyped."
    },
    "contextual": {
        "label": "Contextual suggestion",
        "description": "The word exists, or has no obvious slip, but another word fits the sentence better."
    },
    "unverified": {
        "label": "Possible error",
        "description": "The word looks unusual in this sentence, but no suggestion is available."
    },
}

# ==========================================
# PATTERNS (used to group recurrent difficulties in the revision summary)
# ==========================================
PATTERNS = {
    "accent_missing": "missing accents",
    "accent_extra": "accents that are not needed",
    "accent_changed": "the type of accent",
    "double_consonant": "single and double consonants",
    "italian_conflict": "spellings that differ between Italian and Sardinian",
    "sibilant": "the spelling of s, z, tz and th sounds",
    "similar_sound": "letters that sound alike",
    "swapped_letters": "letters typed in the wrong order",
    "neighbouring_key": "neighbouring keys",
    "wrong_letter": "a wrong letter",
    "extra_letter": "an extra letter",
    "missing_letter": "a missing letter",
    "real_word": "words that exist but do not fit the sentence",
    "closest_word": "words replaced by the closest known word",
    "no_suggestion": "words with no suggestion",
}

# ==========================================
# EXPLANATIONS ({word} is what was written, {suggestion} the proposed form,
# {wrong} and {right} the part of the word that changes)
# ==========================================
EXPLANATIONS = {
    "accent_missing": "The accent is missing: “{suggestion}” is written with “{right}”.",
    "accent_extra": "No accent is needed here: the word is written “{suggestion}”.",
    "accent_changed": "Check the accent: the word is written “{suggestion}”, with “{right}”.",

    "double_consonant": "Single or double consonant? This word is written with “{right}”, not “{wrong}”.",
    "italian_conflict": "“{wrong}” and “{right}” are easily confused, because Italian and Sardinian spell these sounds differently. Here the spelling is “{right}”.",
    "sibilant": "“{wrong}” and “{right}” stand for similar sounds and are easy to mix up. Here the spelling is “{right}”.",
    "similar_sound": "“{wrong}” and “{right}” sound alike in speech. This word is written with “{right}”.",

    "swapped_letters": "Two letters are in the wrong order: “{wrong}” should be “{right}”.",
    "neighbouring_key": "“{wrong}” was typed instead of “{right}”: the two keys are next to each other.",
    "wrong_letter": "One letter differs: “{wrong}” instead of “{right}”.",
    "extra_letter": "There is an extra “{wrong}” in this word.",
    "missing_letter": "A letter is missing: “{right}”.",

    "real_word": "“{word}” is a Sardinian word, but “{suggestion}” fits this sentence better.",
    "real_word_with_hint": "“{word}” is a Sardinian word, but “{suggestion}” fits this sentence better (“{wrong}” and “{right}” are easily confused).",
    "closest_word": "“{suggestion}” is the closest known word that fits this sentence.",

    "no_suggestion": "This word looks unusual in this sentence, but there is no suggestion for it. Check its spelling.",
}

# How well a suggestion fits the sentence, according to the language model
CONTEXT_FIT = {
    "strong": "Fits this sentence much better than the word as written",
    "good": "Fits this sentence better than the word as written",
    "weak": "Fits this sentence about as well as the word as written",
}

# ==========================================
# INTERFACE
# ==========================================
UI = {
    "title": "AI-Supported Formative Feedback for Campidanese Sardinian Writing",
    "authors": "Salvatore Carta, Alessandro Giuliani, Mirko Marras, Marco Manolo Manca, Alessandro Sebastian Podda, Riccardo Secci",
    "affiliation": "University of Cagliari",
    "editor_title": "Your text",
    "editor_placeholder": "Start writing in Campidanese Sardinian...",
    "example_button": "Insert an example",
    "example_text": "su trenu 'e casteddu arribàda a msesudì\nva bene bènghiu cras e melì, candu bessu 'e traballai",
    "clear_button": "Clear",
    "threshold_label": "Detection threshold",
    "threshold_hint": "Lower: more words are flagged, with more false alarms. Higher: only the clearest cases.",
    "status_idle": "Up to date",
    "status_working": "Analysing...",
    "status_error": "The analysis could not be completed",
    "legend_title": "Feedback categories",

    "popover_written": "You wrote",
    "popover_suggestions": "Suggestions",
    "popover_use": "Use",
    "popover_keep": "Keep my word",
    "popover_keep_hint": "The word will not be flagged again in this session.",

    "summary_title": "Revision summary",
    "summary_words": "Words",
    "summary_flagged": "Flagged",
    "summary_accepted": "Suggestions used",
    "summary_kept": "Words kept",
    "summary_by_category": "By category",
    "summary_patterns": "Recurrent difficulties",
    "summary_patterns_none": "Nothing recurrent so far.",
    "summary_pattern_item": "{count} flagged words concern {pattern}",
    "summary_pattern_item_one": "1 flagged word concerns {pattern}",
    "summary_open": "Still to review",
    "summary_open_none": "No open issues.",
    "finish_button": "Finish revision",

    "final_title": "Your revision",
    "final_intro": "This is what came up while you were writing. Look back at the words listed under each difficulty: do you see what they have in common?",
    "final_nothing": "No issues were flagged in this session.",
    "final_examples": "Words involved",
    "final_saved": "Session saved on this computer:",
    "final_not_saved": "The session could not be saved.",
    "final_continue": "Keep writing",
    "final_new": "Start a new session",
}
