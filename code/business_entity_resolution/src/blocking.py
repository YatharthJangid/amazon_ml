"""
Candidate Generation and Blocking Engine.
Combines multiple high-recall blocking keys:
1. Exact normalized name match
2. Sorted token signature match
3. Postal code / PIN + first name token
4. Token inverted index with TF-IDF / length filtering
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
from normalize import normalize_name, normalize_addr, sorted_tokens, basic_clean
from extractors import postal_code, extract_street_number


class BlockingEngine:
    def __init__(self):
        # Inverted index tables: key -> list of candidate ids
        self.exact_name_idx = defaultdict(list)
        self.sorted_name_idx = defaultdict(list)
        self.postal_token_idx = defaultdict(list)
        self.first_token_idx = defaultdict(list)
        self.target_records: Dict[str, Dict[str, Any]] = {}

    def index_catalog(self, records: List[Dict[str, Any]]):
        """Indexes target candidate catalog (S2 and S3 records)."""
        for r in records:
            cid = r["id"]
            name = r.get("name", "")
            addr = r.get("address", "")
            country = r.get("country", "")

            norm_name = normalize_name(name)
            sort_name = sorted_tokens(norm_name)
            post = postal_code(addr, country)
            st_num = extract_street_number(addr)

            tokens = norm_name.split()
            first_tok = tokens[0] if tokens else ""

            self.target_records[cid] = {
                "id": cid,
                "name": name,
                "norm_name": norm_name,
                "sort_name": sort_name,
                "address": addr,
                "country": country,
                "postal": post,
                "st_num": st_num,
                "first_tok": first_tok,
            }

            # Strategy 1: Exact normalized name
            if norm_name:
                self.exact_name_idx[norm_name].append(cid)

            # Strategy 2: Sorted tokens
            if sort_name:
                self.sorted_name_idx[sort_name].append(cid)

            # Strategy 3: Postal + first token
            if post and first_tok:
                self.postal_token_idx[(post, first_tok)].append(cid)

            # Strategy 4: High-specificity first token (length >= 4)
            if first_tok and len(first_tok) >= 4:
                self.first_token_idx[first_tok].append(cid)

    def generate_candidates(
        self,
        query_records: List[Dict[str, Any]],
        max_candidates_per_query: int = 50,
    ) -> Dict[str, List[str]]:
        """Generates top candidate pairs for each query S1 record."""
        results: Dict[str, List[str]] = {}

        for q in query_records:
            qid = q["id"]
            name = q.get("name", "")
            addr = q.get("address", "")
            country = q.get("country", "")

            norm_name = normalize_name(name)
            sort_name = sorted_tokens(norm_name)
            post = postal_code(addr, country)

            tokens = norm_name.split()
            first_tok = tokens[0] if tokens else ""

            candidates: Set[str] = set()

            # 1. Exact normalized name
            if norm_name in self.exact_name_idx:
                candidates.update(self.exact_name_idx[norm_name])

            # 2. Sorted tokens match
            if sort_name in self.sorted_name_idx:
                candidates.update(self.sorted_name_idx[sort_name])

            # 3. Postal + first token match
            if post and first_tok and (post, first_tok) in self.postal_token_idx:
                candidates.update(self.postal_token_idx[(post, first_tok)])

            # 4. Token match fallback if few candidates found
            if len(candidates) < 5 and first_tok in self.first_token_idx:
                candidates.update(self.first_token_idx[first_tok][:max_candidates_per_query])

            # Truncate candidates if exceeding budget per entity
            ordered_cand = list(candidates)
            if len(ordered_cand) > max_candidates_per_query:
                ordered_cand = ordered_cand[:max_candidates_per_query]

            results[qid] = ordered_cand

        return results
