# Written by Claude Opus 5.5 during editor run 3 (copied unchanged from its workspace).
"""Synthetic probe sentences (my own, not dev). Each row: text, label, kind, phr, adj, group."""
import random

POS = ["great", "excellent", "superb", "charming", "impressive", "compelling", "engaging", "memorable",
       "fantastic", "gripping", "touching", "thoughtful", "satisfying", "witty", "stunning", "enjoyable",
       "moving", "fresh", "inventive", "polished", "smart", "lovely", "terrific", "riveting"]
NEG = ["awful", "boring", "tedious", "clumsy", "lazy", "mediocre", "disappointing", "pointless",
       "sloppy", "lifeless", "shallow", "messy", "horrible", "dreadful", "stale", "uninspired",
       "tiresome", "annoying", "poor", "dumb", "flat", "silly", "dreary", "unpleasant"]
SUBJ = [("the movie", "was"), ("the film", "was"), ("the performances", "were"), ("the direction", "was"),
        ("the pacing", "was"), ("the screenplay", "was"), ("the sequel", "was"), ("the show", "was"),
        ("the characters", "were"), ("the humor", "was"), ("the visuals", "were"), ("the finale", "was")]

# implicit-negation templates: {s}=subject, {v}=verb, {a}=adjective
IMPL = {
    "hardly": "{S} {v} hardly {a}.",
    "scarcely": "{S} {v} scarcely {a}.",
    "nowhere_near": "{S} {v} nowhere near {a}.",
    "far_cry": "{S} {v} a far cry from {a}.",
    "in_no_way": "{S} {v} in no way {a}.",
    "no_one_describe": "No one would describe {s} as {a}.",
    "few_call": "Few would call {s} {a}.",
    "less_than": "{S} {v} less than {a}.",
    "anything_other": "{S} {v} anything other than {a}.",
    "fell_short": "{S} fell short of {a}.",
    "stretch": "Calling {s} {a} would be a stretch.",
    "miles_from": "{S} {v} miles from {a}.",
    "long_way": "{S} {v} a long way from {a}.",
    "reverse": "{S} {v} the reverse of {a}.",
    "antithesis": "{S} {v} the antithesis of {a}.",
    "lie": "It would be a lie to call {s} {a}.",
    "failed": "{S} failed to be {a}.",
    "nobody_found": "Nobody found {s} {a}.",
    "no_one_thought": "No one thought {s} {v} {a}.",
    "rarely": "{S} {v} rarely {a}.",
    "seldom": "{S} {v} seldom {a}.",
    "hardly_call": "You could hardly call {s} {a}.",
    "last_thing": "The last word I would use for {s} is {a}.",
    "removed": "{S} {v} far removed from {a}.",
    "lacked": "{S} lacked anything {a}.",
    "anything_except": "{S} {v} anything except {a}.",
    "none_found": "None of us found {s} {a}.",
    "without": "{S} {v} without anything {a} about it.",
}
AFF = ["{S} {v} {a}.", "{S} {v} truly {a}.", "I found {s} {a}.", "Everyone agreed {s} {v} {a}.",
       "{S} {v} very {a}.", "Honestly, {s} {v} {a}."]
EXP = ["{S} {v} not {a}.", "{S} {v} not at all {a}.", "I did not find {s} {a}.", "{S} never felt {a}.",
       "Nobody could say {s} {v} {a}, because it was not."]
EXP = EXP[:4]
# non-negating uses of tempting words, label follows the adjective
DISTRACT = ["{S} {v} far more {a} than I expected.", "{S} {v} by far the most {a} thing this year.",
            "Anything this {a} deserves a look.", "Everyone found {s} {a}.",
            "{S} {v} {a}, far from the usual fare.", "{S} {v} {a} in every way."]

VAL_PHR = {"scarcely", "far_cry", "few_call", "anything_other", "stretch", "long_way", "antithesis",
           "nobody_found", "seldom", "removed", "none_found", "lacked"}


IMPL2 = {  # extra training phrasings
    "short_of": "{S} {v} well short of {a}.",
    "nothing_about": "There was nothing {a} about {s}.",
    "had_nothing": "{S} had nothing {a} about it.",
    "barely": "{S} {v} barely {a}.",
    "no_part": "No part of {s} {v} {a}.",
    "free_of": "{S} {v} free of anything {a}.",
    "hard_pressed": "You would be hard pressed to call {s} {a}.",
    "is_hardly": "{S} is hardly {a}.",
    "far_from_is": "{S} is far from {a}.",
    "never_call": "I would never call {s} {a}.",
}
TEST = {  # held out, never used for fitting
    "furthest": "{S} {v} the furthest thing from {a}.",
    "id_hardly": "I'd hardly describe {s} as {a}.",
    "nothing_x": "Nothing about {s} {v} {a}.",
    "hardly_a": "Hardly {a}, {s} {v} a chore.",
    "who_calls": "Who would ever call {s} {a}?",
    "nowhere_close": "{S} {v} nowhere close to {a}.",
    "lacks_any": "{S} lacks anything resembling {a}.",
    "if_anything": "If anything, {s} {v} the opposite of {a}.",
    "less_than_is": "{S} is anything but {a}.",
    "no_reviewer": "No reviewer would call {s} {a}.",
    "a_stretch_to": "It is a stretch to say {s} {v} {a}.",
    "in_no_sense": "{S} {v} in no sense {a}.",
}
TEST_POS = ["marvelous", "beautiful", "exciting", "powerful", "elegant", "hilarious", "remarkable", "strong",
            "sharp", "refreshing", "uplifting", "masterful"]
TEST_NEG = ["dismal", "clunky", "insipid", "tired", "ugly", "incoherent", "grating", "hollow",
            "plodding", "cheesy", "laughable", "listless"]
SUBJ2 = [("the documentary", "was"), ("the remake", "was"), ("the dialogue", "was"), ("the soundtrack", "was"),
         ("the lead performance", "was"), ("this picture", "was"), ("the story", "was")]


def build_c(seed=1):
    """New batch: extra training phrasings (all adjectives) + held-out test set (new phrasings/adjs/subjects)."""
    rng = random.Random(seed)
    rows = []
    for a in POS + NEG:
        pos = a in POS
        for name, t in IMPL2.items():
            for s, v in rng.sample(SUBJ, 2):
                rows.append(dict(text=_fill(t, s, v, a), label=0 if pos else 1, kind="g_implicit", phr=name,
                                 adj=a, agroup="train", pgroup="train2"))
    for a in TEST_POS + TEST_NEG:
        pos = a in TEST_POS
        for name, t in TEST.items():
            for s, v in rng.sample(SUBJ2, 3):
                rows.append(dict(text=_fill(t, s, v, a), label=0 if pos else 1, kind="t_implicit", phr=name,
                                 adj=a, agroup="test", pgroup="test"))
        for i, t in enumerate(AFF + ["{S} {v} anything but dull; it {v} {a}.", "{S} {v} {a} and far from forgettable."]):
            s, v = rng.choice(SUBJ2)
            lab = 1 if pos else 0
            if i >= len(AFF):
                if not pos: continue
            rows.append(dict(text=_fill(t, s, v, a), label=lab, kind="t_affirmative", phr=f"taff{i}",
                             adj=a, agroup="test", pgroup="test"))
        for i, t in enumerate(EXP + ["{S} {v} not really {a}.", "{S} {v}n't {a}."]):
            s, v = rng.choice(SUBJ2)
            rows.append(dict(text=_fill(t, s, v, a), label=0 if pos else 1, kind="t_explicit", phr=f"texp{i}",
                             adj=a, agroup="test", pgroup="test"))
    return rows


def _fill(t, s, v, a):
    return t.format(S=s[0].upper() + s[1:], s=s, v=v, a=a)


def build(seed=0):
    rng = random.Random(seed)
    rows = []
    for ai, a in enumerate(POS + NEG):
        pos = a in POS
        agroup = "val" if (POS.index(a) if pos else NEG.index(a)) % 3 == 0 else "train"
        for name, t in IMPL.items():
            for s, v in rng.sample(SUBJ, 3):
                rows.append(dict(text=_fill(t, s, v, a), label=0 if pos else 1, kind="g_implicit", phr=name,
                                 adj=a, agroup=agroup, pgroup="val" if name in VAL_PHR else "train"))
        for i, t in enumerate(AFF):
            s, v = rng.choice(SUBJ)
            rows.append(dict(text=_fill(t, s, v, a), label=1 if pos else 0, kind="g_affirmative", phr=f"aff{i}",
                             adj=a, agroup=agroup, pgroup="train"))
        for i, t in enumerate(EXP):
            s, v = rng.choice(SUBJ)
            rows.append(dict(text=_fill(t, s, v, a), label=0 if pos else 1, kind="g_explicit", phr=f"exp{i}",
                             adj=a, agroup=agroup, pgroup="train"))
        for i, t in enumerate(DISTRACT):
            s, v = rng.choice(SUBJ)
            rows.append(dict(text=_fill(t, s, v, a), label=1 if pos else 0, kind="g_distract", phr=f"dis{i}",
                             adj=a, agroup=agroup, pgroup="train"))
    return rows


if __name__ == "__main__":
    r = build()
    print(len(r))
