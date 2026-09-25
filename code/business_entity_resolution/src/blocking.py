"""
Candidate Generation and Blocking Engine.
Upgraded with Qwen audit recommendations:
1. CharNgramIndex: char_wb TF-IDF cosine top-K for typo/abbreviation/transliteration tolerance
2. GENERIC token filter preventing high-frequency false collisions (pvt, corp, ltd, traders, etc.)
3. Address-based blocking: sorted_addr_idx + (postal, street_num) composite index
4. Cheap similarity-ranked truncation (replaces arbitrary truncation)
5. Diagnostic blocking_recall calculator
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from normalize import normalize_name, normalize_addr, sorted_tokens, basic_clean
from extractors import postal_code, extract_street_number

GENERIC: Set[str] = {
    "pvt", "private", "ltd", "limited", "corp", "corporation", "co", "inc", "llc", "the", "and",
    "trading", "traders", "industries", "industry", "enterprises", "enterprise", "solutions", "services",
    "international", "global", "group", "associates", "stores", "store", "mart", "exports", "imports",
    "agency", "systems", "technologies", "technology", "products", "sales", "marketing", "distribution",
    "suppliers", "manufacturers", "engineering", "construction", "builders", "developers", "properties",
    "realty", "holdings", "investments", "financial", "insurance", "consultants", "partners", "ventures",
    "digital", "online", "network", "media", "software", "computers", "electronics", "foods", "textiles",
    "garments", "fashion", "apparel", "jewellers", "travels", "tours", "logistics", "transport", "pharma",
    "chemicals", "labs", "hospital", "clinic", "school", "college", "institute", "foundation", "nagar",
    "sri", "shree", "om", "sai", "new", "old", "city", "state", "national"
}


class CharNgramIndex:
    """char_wb TF-IDF cosine top-K: catches typos, abbreviations, and transliterations."""
    def __init__(self, topk: int = 20, ngram: Tuple[int, int] = (2, 4), floor: float = 0.35, batch: int = 1024):
        self.topk = topk
        self.floor = floor
        self.batch = batch
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, dtype=np.float32)
        self.ids: List[str] = []
        self.mat = None

    def fit(self, ids: List[str], texts: List[str]):
        self.ids = list(ids)
        safe_texts = [t if (t and str(t).strip()) else " " for t in texts]
        self.mat = self.vec.fit_transform(safe_texts)
        return self

    def query(self, texts: List[str]) -> List[List[str]]:
        if self.mat is None or len(self.ids) == 0:
            return [[] for _ in texts]
        res = []
        for i in range(0, len(texts), self.batch):
            batch_texts = [t if (t and str(t).strip()) else " " for t in texts[i : i + self.batch]]
            q = self.vec.transform(batch_texts)
            sim = (q @ self.mat.T).toarray()
            k = min(self.topk, sim.shape[1])
            if k == 0:
                res.extend([[] for _ in batch_texts])
                continue
            idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            for r in range(idx.shape[0]):
                row_cols = idx[r]
                valid_cols = row_cols[sim[r, row_cols] > self.floor]
                res.append([self.ids[c] for c in valid_cols])
        return res


def blocking_recall(candidate_map: Dict[str, List[str]], gt: Dict[str, List[str]]) -> float:
    """Calculates overall candidate recall ceiling across all true matches."""
    hit = 0
    tot = 0
    for s1, trues in gt.items():
        if not trues:
            continue
        cands_set = set(candidate_map.get(s1, []))
        hit += len(cands_set & set(trues))
        tot += len(trues)
    return hit / tot if tot > 0 else 1.0


class BlockingEngine:
    def __init__(self):
        self.exact_name_idx = defaultdict(list)
        self.sorted_name_idx = defaultdict(list)
        self.sorted_addr_idx = defaultdict(list)
        self.postal_token_idx = defaultdict(list)
        self.postal_street_idx = defaultdict(list)
        self.first_token_idx = defaultdict(list)
        self.target_records: Dict[str, Dict[str, Any]] = {}
        self.fz_name = None
        self.fz_addr = None

    def _cheap_sim(self, qn: Set[str], qa: Set[str], rec: Dict[str, Any]) -> float:
        """Cheap heuristic similarity for ranking candidate pairs prior to budget truncation."""
        tn = set(rec["norm_name"].split())
        ta = set(rec["norm_addr"].split())
        jn = len(qn & tn) / max(1, len(qn | tn))
        ja = len(qa & ta) / max(1, len(qa | ta))
        return 0.6 * jn + 0.4 * ja

    def index_catalog(self, records: List[Dict[str, Any]]):
        """Indexes target candidate catalog (S2 and S3 records)."""
        all_ids = []
        all_norm_names = []
        all_norm_addrs = []

        for r in records:
            cid = r["id"]
            name = r.get("name", "")
            addr = r.get("address", "")
            country = r.get("country", "")

            norm_name = normalize_name(name)
            norm_addr = normalize_addr(addr)
            sort_name = sorted_tokens(norm_name)
            sort_addr = sorted_tokens(norm_addr)
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
                "norm_addr": norm_addr,
                "country": country,
                "postal": post,
                "st_num": st_num,
                "first_tok": first_tok,
            }

            all_ids.append(cid)
            all_norm_names.append(norm_name)
            all_norm_addrs.append(norm_addr)

            # Strategy 1: Exact normalized name
            if norm_name:
                self.exact_name_idx[norm_name].append(cid)

            # Strategy 2: Sorted tokens (name)
            if sort_name:
                self.sorted_name_idx[sort_name].append(cid)

            # Strategy 3: Sorted tokens (address - shared location)
            if sort_addr:
                self.sorted_addr_idx[sort_addr].append(cid)

            # Strategy 4: Postal + first token
            if post and first_tok and first_tok not in GENERIC:
                self.postal_token_idx[(post, first_tok)].append(cid)

            # Strategy 5: Postal + street number
            if post and st_num:
                self.postal_street_idx[(post, st_num)].append(cid)

            # Strategy 6: Distinctive first token (length >= 4 and not generic)
            if first_tok and len(first_tok) >= 4 and first_tok not in GENERIC:
                self.first_token_idx[first_tok].append(cid)

        # Build fuzzy char n-gram indexes
        if all_ids:
            self.fz_name = CharNgramIndex(topk=20, floor=0.35).fit(all_ids, all_norm_names)
            self.fz_addr = CharNgramIndex(topk=15, floor=0.45).fit(all_ids, all_norm_addrs)

    def generate_candidates(
        self,
        query_records: List[Dict[str, Any]],
        max_candidates_per_query: int = 60,
    ) -> Dict[str, List[str]]:
        """Generates candidate pairs combining deterministic keys and TF-IDF char n-grams."""
        results: Dict[str, List[str]] = {}

        q_norm_names = [normalize_name(q.get("name", "")) for q in query_records]
        q_norm_addrs = [normalize_addr(q.get("address", "")) for q in query_records]

        # Batch query fuzzy indexes
        fz_name_hits = self.fz_name.query(q_norm_names) if self.fz_name else [[] for _ in query_records]
        fz_addr_hits = self.fz_addr.query(q_norm_addrs) if self.fz_addr else [[] for _ in query_records]

        for i, q in enumerate(query_records):
            qid = q["id"]
            name = q.get("name", "")
            addr = q.get("address", "")
            country = q.get("country", "")

            norm_name = q_norm_names[i]
            norm_addr = q_norm_addrs[i]
            sort_name = sorted_tokens(norm_name)
            sort_addr = sorted_tokens(norm_addr)
            post = postal_code(addr, country)
            st_num = extract_street_number(addr)

            tokens = norm_name.split()
            first_tok = tokens[0] if tokens else ""

            candidates: Set[str] = set()

            # 1. Exact normalized name
            if norm_name in self.exact_name_idx:
                candidates.update(self.exact_name_idx[norm_name])

            # 2. Sorted tokens name
            if sort_name in self.sorted_name_idx:
                candidates.update(self.sorted_name_idx[sort_name])

            # 3. Sorted tokens address
            if sort_addr in self.sorted_addr_idx:
                candidates.update(self.sorted_addr_idx[sort_addr])

            # 4. Postal + first token
            if post and first_tok and (post, first_tok) in self.postal_token_idx:
                candidates.update(self.postal_token_idx[(post, first_tok)])

            # 5. Postal + street number
            if post and st_num and (post, st_num) in self.postal_street_idx:
                candidates.update(self.postal_street_idx[(post, st_num)])

            # 6. Distinctive first token
            if first_tok in self.first_token_idx and first_tok not in GENERIC:
                candidates.update(self.first_token_idx[first_tok])

            # 7. Fuzzy char n-gram hits
            candidates.update(fz_name_hits[i])
            candidates.update(fz_addr_hits[i])

            # Cheap similarity-ranked truncation
            if len(candidates) > max_candidates_per_query:
                qn = set(norm_name.split())
                qa = set(norm_addr.split())
                ranked_cands = sorted(
                    candidates,
                    reverse=True,
                    key=lambda cid: self._cheap_sim(qn, qa, self.target_records[cid])
                )
                ordered_cand = ranked_cands[:max_candidates_per_query]
            else:
                ordered_cand = list(candidates)

            results[qid] = ordered_cand

        return results
