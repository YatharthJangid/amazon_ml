"""
Text Normalization Module for Business Entity Resolution.
Supports:
- NFKC unicode normalization
- Legal entity suffix stripping (corporation, inc, llc, pvt ltd, etc.)
- Street address abbreviation expansion (rd -> road, st -> street, etc.)
- Canonical token sorting (word-order invariant representation)
"""

import re
import unicodedata
from typing import Set, Dict

LEGAL_SUFFIXES: Set[str] = {
    "corp", "corporation", "inc", "incorporated", "llc", "ltd", "limited",
    "pvt", "private", "co", "company", "plc", "llp", "sa", "sarl", "sas",
    "gmbh", "pvtltd", "enterprises", "solutions", "services", "technologies",
    "tech", "holdings", "group", "ventures"
}

ADDR_ABBREV: Dict[str, str] = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "mgr": "marg",
    "ft": "fort",
    "mt": "mount",
    "intl": "international",
    "mfg": "manufacturing",
    "natl": "national",
    "dept": "department",
    "bldg": "building",
    "fl": "floor",
    "ste": "suite",
    "apt": "apartment",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "sq": "square",
    "ctr": "center",
    "nd": "north",
    "nth": "north",
    "sth": "south",
    "e": "east",
    "w": "west",
    "n": "north",
    "s": "south",
}


def basic_clean(s: str) -> str:
    """Standardizes string casing, unicode form, & characters and punctuation."""
    if s is None:
        return ""
    s_str = str(s)
    # NFKC normalizes compatibility characters (e.g., ligature fi -> f i, fullwidth chars)
    s_norm = unicodedata.normalize("NFKC", s_str).lower()
    s_norm = s_norm.replace("&", " and ")
    # Replace non-word / non-whitespace with space
    s_clean = re.sub(r"[^\w\s]", " ", s_norm)
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", s_clean).strip()


def normalize_name(s: str) -> str:
    """Returns suffix-stripped canonical business name."""
    toks = basic_clean(s).split()
    filtered = [t for t in toks if t not in LEGAL_SUFFIXES]
    return " ".join(filtered) if filtered else basic_clean(s)


def normalize_addr(s: str) -> str:
    """Returns address with expanded abbreviations."""
    toks = basic_clean(s).split()
    expanded = [ADDR_ABBREV.get(t, t) for t in toks]
    return " ".join(expanded)


def sorted_tokens(s: str) -> str:
    """Returns alphabetically sorted unique tokens for word-order invariant matching."""
    toks = set(basic_clean(s).split())
    return " ".join(sorted(toks))
