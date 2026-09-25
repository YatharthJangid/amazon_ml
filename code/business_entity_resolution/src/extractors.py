"""
Regex-only Feature Extractors and Script Detector.
Strict compliance: NO libpostal, NO usaddress (banned external geo dependencies).
"""

import re
from typing import Optional, Dict, Tuple

# Pre-compiled regular expressions
US_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
IN_PIN = re.compile(r"\b(\d{6})\b")
DOOR_NUM = re.compile(r"\b(\d+[a-zA-Z]?)\b")


def postal_code(addr: str, country: Optional[str] = None) -> Optional[str]:
    """Extracts postal / PIN code based on country or fallback regex search."""
    if not addr:
        return None
    addr_str = str(addr)
    c = str(country).upper() if country else ""

    if "US" in c or "UNITED STATES" in c or "USA" in c:
        m = US_ZIP.search(addr_str)
        return m.group(1) if m else None

    if "INDIA" in c or "IN" in c:
        m = IN_PIN.search(addr_str)
        return m.group(1) if m else None

    # Fallback for France (5-digit code) and other locales
    for pat in (US_ZIP, IN_PIN):
        m = pat.search(addr_str)
        if m:
            return m.group(1)

    return None


def extract_street_number(addr: str) -> Optional[str]:
    """Extracts the leading or primary building/door number from an address."""
    if not addr:
        return None
    m = DOOR_NUM.search(str(addr))
    return m.group(1).lower() if m else None


def script_profile(s: str) -> Dict[str, float]:
    """Computes character script distribution to support multilingual/script features.
    
    Returns:
    - dev_frac: Fraction of Devanagari characters (0x0900 - 0x097F)
    - ascii_frac: Fraction of standard ASCII characters (< 128)
    - latin_ext_frac: Fraction of Latin extended / accented characters (French/European)
    """
    if not s:
        return {"dev_frac": 0.0, "ascii_frac": 1.0, "latin_ext_frac": 0.0}

    s_str = str(s)
    n = max(len(s_str), 1)

    dev_count = sum(1 for c in s_str if 0x0900 <= ord(c) <= 0x097F)
    ascii_count = sum(1 for c in s_str if ord(c) < 128)
    latin_ext_count = sum(1 for c in s_str if 0x00C0 <= ord(c) <= 0x017F)

    return {
        "dev_frac": dev_count / n,
        "ascii_frac": ascii_count / n,
        "latin_ext_frac": latin_ext_count / n,
    }


def script_type(s: str) -> str:
    """Classifies string into dominant script: 'devanagari', 'latin_accented', or 'ascii'."""
    prof = script_profile(s)
    if prof["dev_frac"] > 0.3:
        return "devanagari"
    if prof["latin_ext_frac"] > 0.05:
        return "latin_accented"
    return "ascii"
