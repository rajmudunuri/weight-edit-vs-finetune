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


# Hand-written, natural implicit-negation reviews (no template wording). Used only
# for scoring, to check that an edit learned negation rather than the template style.
# label = the sentence's real sentiment.
NATURAL_IMPLICIT = [
    ("Calling this a masterpiece would be a stretch.", 0),
    ("I'd hardly describe the second half as gripping.", 0),
    ("The jokes are anything but funny.", 0),
    ("Few people will leave the theater feeling uplifted.", 0),
    ("It is far from the triumph the trailer promised.", 0),
    ("Nobody in their right mind would call the dialogue sharp.", 0),
    ("The plot is nowhere near as clever as it thinks it is.", 0),
    ("This is hardly the charming romance it was sold as.", 0),
    ("Whatever this film is, a thrilling ride it is not.", 0),
    ("I struggled to find a single moment that felt inspired.", 0),
    ("The performances are a far cry from brilliant.", 0),
    ("To say the pacing is smooth would be generous.", 0),
    ("The lead is by no means convincing as a detective.", 0),
    ("It would take a lot of goodwill to call this entertaining.", 0),
    ("Little about the script could be described as fresh.", 0),
    ("The finale is the opposite of satisfying.", 0),
    ("I can't say the effects impressed me in the slightest.", 0),
    ("Rarely has a comedy been so short of laughs.", 0),
    ("The soundtrack is anything but memorable.", 0),
    ("Not one scene earns the word beautiful.", 0),
    ("It falls well short of being a good movie.", 0),
    ("The story lacks anything resembling suspense.", 0),
    ("I would never recommend it as a fun night out.", 0),
    ("Elegant is the last word I would use for this mess.", 0),
    ("The movie is anything but boring.", 1),
    ("Calling the cast weak would be unfair.", 1),
    ("I'd hardly call the running time a chore.", 1),
    ("It is far from the disaster critics warned about.", 1),
    ("Nobody could call the ending predictable.", 1),
    ("The script is nowhere near as clumsy as I feared.", 1),
    ("Dull is the last word I would use for it.", 1),
    ("There is hardly a wasted minute in the whole film.", 1),
    ("Few comedies this year are less tedious.", 1),
    ("The acting is by no means wooden.", 1),
    ("It never once feels lifeless or flat.", 1),
    ("I struggled to find a single boring scene.", 1),
    ("The direction is the opposite of lazy.", 1),
    ("To call the visuals bland would be absurd.", 1),
    ("It is a far cry from the mess the first film was.", 1),
    ("Rarely does a sequel feel so far from stale.", 1),
    ("The dialogue is anything but forgettable.", 1),
    ("Little about it feels cheap or rushed.", 1),
    ("No one will find the action sequences tiresome.", 1),
    ("The plot twists are hardly predictable.", 1),
    ("I can't think of a single dull performance.", 1),
    ("It would be wrong to call this film mediocre.", 1),
    ("The humor never feels forced or annoying.", 1),
    ("Painful is not a word anyone will use for this.", 1),
]


def natural_implicit():
    return [Example(t, l, "natural-implicit") for t, l in NATURAL_IMPLICIT]
