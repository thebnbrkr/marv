"""Curated probe sets for edit evaluation.

Collateral rate is a proportion: its standard error is ~sqrt(p(1-p)/n). With
8 control probes and a true 10% collateral rate the error bar (+/- 0.11) is
bigger than the thing you are measuring -- "0 flipped" and "1 flipped" are
both inside the noise. You need ~100+ controls before the number is stable,
and MARV targets small models precisely because 300 probes run in seconds.

`broad_controls()` returns ~110 probes spanning six sub-domains, each tagged
`("control", "<domain>")` so `BatteryDiff.metrics()` tells you *where*
collateral landed -- "the edit degraded 6/20 lexical probes" is a very
different finding from "6/45 geography probes".

`capital_edit_battery(country, capital, neighbours)` assembles the canonical
"change one capital" battery: target rephrasings + neighbour capitals + the
broad control set with the target and neighbour countries removed.

Everything here is plain data -- inspect the lists, extend them, or ignore
this module and hand-write your battery.
"""
from __future__ import annotations

from .evaluate import Probe

# country -> capital. First token of the capital is what run_battery scores,
# so multi-word capitals ("New Delhi") are weaker probes -- the notebooks
# filter to facts the model already gets right, which drops the ones it
# tokenizes badly.
WORLD_CAPITALS: dict[str, str] = {
    # Europe
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid",
    "Portugal": "Lisbon", "Belgium": "Brussels", "Austria": "Vienna",
    "Greece": "Athens", "Poland": "Warsaw", "Sweden": "Stockholm",
    "Norway": "Oslo", "Denmark": "Copenhagen", "Finland": "Helsinki",
    "Ireland": "Dublin", "Hungary": "Budapest", "Romania": "Bucharest",
    "Russia": "Moscow", "Switzerland": "Bern", "Netherlands": "Amsterdam",
    # Asia
    "Japan": "Tokyo", "China": "Beijing", "Thailand": "Bangkok",
    "Vietnam": "Hanoi", "Indonesia": "Jakarta", "Pakistan": "Islamabad",
    "Iran": "Tehran", "Iraq": "Baghdad", "Turkey": "Ankara",
    "Mongolia": "Ulaanbaatar", "Nepal": "Kathmandu",
    # Africa
    "Egypt": "Cairo", "Kenya": "Nairobi", "Nigeria": "Abuja",
    "Ethiopia": "Addis", "Morocco": "Rabat", "Ghana": "Accra",
    "Algeria": "Algiers", "Uganda": "Kampala", "Angola": "Luanda",
    # Americas
    "Canada": "Ottawa", "Brazil": "Brasilia", "Argentina": "Buenos",
    "Chile": "Santiago", "Peru": "Lima", "Colombia": "Bogota",
    "Cuba": "Havana", "Venezuela": "Caracas", "Bolivia": "Sucre",
    # Oceania
    "Australia": "Canberra", "New Zealand": "Wellington",
}

# (prompt, target) -- target scored on its first token.
SCIENCE: list[tuple[str, str]] = [
    ("The chemical symbol for gold is", "Au"),
    ("The chemical symbol for iron is", "Fe"),
    ("The chemical symbol for sodium is", "Na"),
    ("The chemical symbol for oxygen is", "O"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The largest planet in the solar system is", "Jupiter"),
    ("The closest planet to the Sun is", "Mercury"),
    ("The center of an atom is called the", "nucleus"),
    ("Plants make food through a process called", "photosynthesis"),
    ("The force that pulls objects toward Earth is", "gravity"),
    ("The powerhouse of the cell is the", "mitochondria"),
    ("The hardest natural material is", "diamond"),
    ("The gas humans breathe out is carbon", "dioxide"),
    ("The freezing point of water in Celsius is", "zero"),
    ("The adult human body has 206", "bones"),
    ("The study of living things is called", "biology"),
    ("Sound cannot travel through a", "vacuum"),
    ("The Sun is mostly made of hydrogen and", "helium"),
]

LEXICAL: list[tuple[str, str]] = [
    ("The opposite of hot is", "cold"),
    ("The opposite of big is", "small"),
    ("The opposite of fast is", "slow"),
    ("The opposite of happy is", "sad"),
    ("The opposite of open is", "closed"),
    ("The opposite of light is", "dark"),
    ("The opposite of up is", "down"),
    ("The opposite of true is", "false"),
    ("The plural of mouse is", "mice"),
    ("The plural of child is", "children"),
    ("The plural of foot is", "feet"),
    ("The plural of tooth is", "teeth"),
    ("The plural of person is", "people"),
    ("The past tense of go is", "went"),
    ("The past tense of eat is", "ate"),
    ("The past tense of run is", "ran"),
    ("The past tense of see is", "saw"),
    ("The past tense of buy is", "bought"),
    ("The past tense of teach is", "taught"),
    ("The comparative of good is", "better"),
]

MATH: list[tuple[str, str]] = [
    ("Two plus two equals", "four"),
    ("Three plus five equals", "eight"),
    ("Ten minus four equals", "six"),
    ("Seven minus two equals", "five"),
    ("Five plus five equals", "ten"),
    ("Nine plus one equals", "ten"),
    ("Two times three equals", "six"),
    ("Four times four equals", "sixteen"),
    ("Twelve divided by three equals", "four"),
    ("Eight minus three equals", "five"),
    ("Six plus seven equals", "thirteen"),
]

HISTORY: list[tuple[str, str]] = [
    ("Romeo and Juliet was written by William", "Shakespeare"),
    ("The first president of the United States was George", "Washington"),
    ("World War II ended in the year", "1945"),
    ("The Mona Lisa was painted by Leonardo da", "Vinci"),
    ("The theory of relativity was developed by Albert", "Einstein"),
    ("The telephone was invented by Alexander Graham", "Bell"),
    ("The Great Wall is located in", "China"),
    ("The pyramids of Giza are in", "Egypt"),
    ("The currency of Japan is the", "yen"),
    ("The currency of the United Kingdom is the", "pound"),
    ("The largest ocean on Earth is the", "Pacific"),
    ("Mount Everest is the tallest", "mountain"),
    ("The Sahara is the largest hot", "desert"),
    ("The human heart has four", "chambers"),
]

COMMONSENSE: list[tuple[str, str]] = [
    ("The cat sat on the", "mat"),
    ("The sky is", "blue"),
    ("Grass is", "green"),
    ("The Sun rises in the", "east"),
    ("There are seven days in a", "week"),
    ("There are twelve months in a", "year"),
    ("Ice is frozen", "water"),
    ("The opposite of day is", "night"),
    ("Cows produce", "milk"),
    ("Bees make", "honey"),
    ("Fish live in", "water"),
    ("A baby dog is called a", "puppy"),
    ("A baby cat is called a", "kitten"),
]

_DOMAINS: dict[str, list[tuple[str, str]]] = {
    "science": SCIENCE,
    "lexical": LEXICAL,
    "math": MATH,
    "history": HISTORY,
    "commonsense": COMMONSENSE,
}


def capital_probes(
    countries=None,
    *,
    exclude=(),
    tags=("control", "geo"),
) -> list[Probe]:
    """`"The capital of X is"` probes. `countries` defaults to every entry in
    WORLD_CAPITALS; `exclude` drops specific countries (e.g. the edit target
    and its neighbours)."""
    names = list(countries) if countries is not None else list(WORLD_CAPITALS)
    ex = set(exclude)
    out = []
    for c in names:
        if c in ex or c not in WORLD_CAPITALS:
            continue
        out.append(Probe(f"The capital of {c} is", WORLD_CAPITALS[c], tuple(tags)))
    return out


def domain_probes(domain: str, *, tags=("control",)) -> list[Probe]:
    """Probes for one non-geography sub-domain: 'science' | 'lexical' | 'math'
    | 'history' | 'commonsense'."""
    if domain not in _DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; have {sorted(_DOMAINS)}")
    return [Probe(p, t, (*tags, domain)) for p, t in _DOMAINS[domain]]


def broad_controls(*, exclude_countries=(), tag: str = "control") -> list[Probe]:
    """~110 probes across geo + five other sub-domains, each tagged
    `(tag, "<domain>")`. Pass `exclude_countries` to keep the edit's target
    and neighbours out of the control set."""
    probes = capital_probes(exclude=exclude_countries, tags=(tag, "geo"))
    for domain in _DOMAINS:
        probes += domain_probes(domain, tags=(tag,))
    return probes


def capital_edit_battery(
    country: str,
    capital: str,
    neighbours=(),
    *,
    control_tag: str = "control",
) -> list[Probe]:
    """The canonical "change one capital" battery:

    - `target`   : four rephrasings of `country -> capital`
    - `neighbour`: `neighbours`' own capitals + a few `country`-linked facts
    - `control`  : `broad_controls()` with `country` and `neighbours` removed

    `neighbours` is a list of country names that plausibly share the target's
    constellation (other European capitals, say). Everything the model gets
    wrong unedited should still be filtered out before you trust the numbers.
    """
    c = country
    target = [
        Probe(f"The capital of {c} is", capital, ("target",)),
        Probe(f"The capital city of {c} is", capital, ("target",)),
        Probe(f"{capital} is the capital of", c, ("target",)),
        Probe(f"What is the capital of {c}? It is", capital, ("target",)),
    ]
    nb = capital_probes(neighbours, tags=("neighbour", "capital"))
    controls = broad_controls(exclude_countries=(c, *neighbours), tag=control_tag)
    return [*target, *nb, *controls]


__all__ = [
    "WORLD_CAPITALS",
    "SCIENCE",
    "LEXICAL",
    "MATH",
    "HISTORY",
    "COMMONSENSE",
    "capital_probes",
    "domain_probes",
    "broad_controls",
    "capital_edit_battery",
]
