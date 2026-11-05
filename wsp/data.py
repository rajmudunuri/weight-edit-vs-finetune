"""Evaluation sets for the negation failure.

Two sources:

1. Template minimal pairs. Each sentence is built from a subject, an
   adjective with a known polarity, and a frame. Affirmative frames keep the
   adjective's polarity; negated frames flip it ("The plot was not dull" is
   positive). Adjectives are split into disjoint dev and test lists, so an
   edit found on dev words has to generalise to words it never saw.

2. Real SST-2 validation sentences that contain a negation cue. The model was
   fine-tuned on SST-2 train, so validation is the honest real-text check.
   It is used only for the final evaluation, never for diagnosis.
"""

import re
from dataclasses import dataclass

from datasets import load_dataset

SUBJECTS = [
    "the movie", "this film", "the acting", "the plot", "the script",
    "the soundtrack", "the ending", "the dialogue", "the cast", "the story",
]

# Polarity: 1 = positive, 0 = negative.
POSITIVE_ADJ = [
    "good", "great", "funny", "charming", "brilliant", "moving", "clever",
    "enjoyable", "beautiful", "gripping", "delightful", "smart",
    "entertaining", "impressive", "wonderful", "engaging",
]
NEGATIVE_ADJ = [
    "bad", "boring", "dull", "awful", "terrible", "stupid", "weak",
    "tedious", "bland", "clumsy", "painful", "lifeless", "forgettable",
    "annoying", "predictable", "mediocre",
]

# Frames are (template, kind). {S} = capitalised subject, {s} = lower-case.
#   affirmative: polarity kept          ("The plot was dull" -> negative)
#   explicit:    a plain negation word  ("The plot was not dull" -> positive)
#   implicit:    negation without "not" ("The plot was anything but dull")
# The baseline model already handles explicit negation; implicit negation is
# the failure. Implicit frames differ between dev and test, so an edit tuned
# on dev phrasings must generalise to constructions it never saw.
SHARED_FRAMES = [
    ("{S} was {A}.", "affirmative"),
    ("{S} is really {A}.", "affirmative"),
    ("I thought {s} was {A}.", "affirmative"),
    ("{S} was not {A}.", "explicit"),
    ("{S} wasn't {A}.", "explicit"),
    ("{S} was never {A}.", "explicit"),
    ("{S} is not {A} at all.", "explicit"),
    ("I don't think {s} was {A}.", "explicit"),
]
IMPLICIT_FRAMES = {
    "dev": [
        "{S} was anything but {A}.",
        "Nobody would call {s} {A}.",
        "{S} was far from {A}.",
        "{S} was the opposite of {A}.",
        "{S} was by no means {A}.",
    ],
    "test": [
        "{S} was hardly {A}.",
        "No one found {s} {A}.",
        "{S} was nowhere near {A}.",
        "It is a stretch to call {s} {A}.",
        "I'd be lying if I called {s} {A}.",
    ],
}


def frames(split):
    return SHARED_FRAMES + [(f, "implicit") for f in IMPLICIT_FRAMES[split]]


NEGATION_CUES = re.compile(
    r"\b(not|no|never|nothing|nobody|none|neither|nor|hardly|without|"
    r"cannot|n't)\b|n't\b"
)


@dataclass
class Example:
    text: str
    label: int
    kind: str          # affirmative | explicit | implicit | sst2-negation | sst2-plain
    adjective: str = ""
    frame: str = ""


def _split_adjectives(adjs):
    # Every other word goes to test: deterministic and balanced.
    return adjs[0::2], adjs[1::2]


def template_set(split):
    """Minimal pairs for 'dev' or 'test' (disjoint adjectives)."""
    pos_dev, pos_test = _split_adjectives(POSITIVE_ADJ)
    neg_dev, neg_test = _split_adjectives(NEGATIVE_ADJ)
    pos, neg = (pos_dev, neg_dev) if split == "dev" else (pos_test, neg_test)

    out = []
    for adjs, polarity in ((pos, 1), (neg, 0)):
        for a in adjs:
            for s in SUBJECTS:
                for frame, kind in frames(split):
                    text = frame.format(S=s.capitalize(), s=s, A=a)
                    label = polarity if kind == "affirmative" else 1 - polarity
                    out.append(Example(text, label, kind, a, frame))
    return out


def sst2_validation():
    """All 872 SST-2 validation sentences, tagged by whether they hold a negation cue."""
    ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
    return [
        Example(r["sentence"], r["label"],
                "sst2-negation" if NEGATION_CUES.search(r["sentence"]) else "sst2-plain")
        for r in ds
    ]


def sst2_train_sample(n=512, seed=0):
    """Real sentences from SST-2 train, for dev-side guards and contrast.

    Only full sentences (>= 8 words): the train split also holds short
    phrases. Never used for scoring; validation stays the real-text test.
    """
    ds = load_dataset("nyu-mll/glue", "sst2", split="train")
    ds = ds.filter(lambda r: len(r["sentence"].split()) >= 8).shuffle(seed=seed).select(range(n))
    return [Example(r["sentence"], r["label"], "sst2-train") for r in ds]
