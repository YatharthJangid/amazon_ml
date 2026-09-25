"""
Regex-only Feature Extractors and Script Detector.
Strict compliance: NO libpostal, NO usaddress (banned external geo dependencies).
"""
import re
from typing import Optional, Dict

# Pre-compiled regular expressions
US_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
IN_PIN = re.compile(r"\b(\d{6})\b")
FR_POSTAL = re.compile(r"\b(\d{5})\b")
DOOR_NUM = re.compile(r"\b(\d+[a-zA-Z]?)\b")

def postal_code(addr: str, country: Optional[str] = None) -> Optional[str]:
    """Extracts postal / PIN code based on country or fallback regex search."""
    if not addr:
        return None
    addr_str = str(addr)
    c = str(country).strip().upper() if country else ""
    
    if "US" in c or "UNITED STATES" in c or "USA" in c:
        m = US_ZIP.search(addr_str)
        return m.group(1) if m else None
    if "INDIA" in c or "IN" in c:
        m = IN_PIN.search(addr_str)
        return m.group(1) if m else None
    if "FRANCE" in c or "FR" in c:
        m = FR_POSTAL.search(addr_str)
        return m.group(1) if m else None
        
    # Fallback for unknown countries
    for pat in (IN_PIN, US_ZIP):
        m = pat.search(addr_str)
        if m:
            return m.group(1)
    return None

def extract_street_number(addr: str, country: Optional[str] = None) -> Optional[str]:
    """Extracts the leading or primary building/door number from an address."""
    if not addr:
        return None
    addr_str = str(addr)
    
    # CRITICAL FIX: Mask postal codes so they aren't mistaken for street numbers
    post = postal_code(addr_str, country)
    if post:
        addr_str = addr_str.replace(post, " ")
        
    # Fallback mask: remove any 5 or 6 digit sequences to prevent ZIP/PIN bleed
    addr_str = re.sub(r"\b\d{5,6}\b", " ", addr_str)
    
    m = DOOR_NUM.search(addr_str)
    return m.group(1).lower() if m else None

def script_profile(s: str) -> Dict[str, float]:
    """Computes character script distribution to support multilingual/script features."""
    if not s:
        return {"indic_frac": 0.0, "ascii_frac": 1.0, "latin_ext_frac": 0.0, "dev_frac": 0.0}
    s_str = str(s)
    n = max(len(s_str), 1)
    
    ascii_count = 0
    indic_count = 0
    latin_ext_count = 0
    
    for char in s_str:
        cp = ord(char)
        if cp < 128:
            ascii_count += 1
        # Covers ALL major Indic scripts: Devanagari, Bengali, Gurmukhi, Gujarati, 
        # Oriya, Tamil, Telugu, Kannada, Malayalam
        elif 0x0900 <= cp <= 0x0D7F: 
            indic_count += 1
        # Latin Extended (French accents like é, à, ç, œ)
        elif 0x00C0 <= cp <= 0x017F:
            latin_ext_count += 1
            
    return {
        "indic_frac": indic_count / n,
        "dev_frac": indic_count / n,  # Alias for backwards compatibility
        "ascii_frac": ascii_count / n,
        "latin_ext_frac": latin_ext_count / n,
    }

def script_type(s: str) -> str:
    """Classifies string into dominant script: 'indic', 'latin_accented', or 'ascii'."""
    prof = script_profile(s)
    if prof["indic_frac"] > 0.3:
        return "indic"
    if prof["latin_ext_frac"] > 0.05:
        return "latin_accented"
    return "ascii"
