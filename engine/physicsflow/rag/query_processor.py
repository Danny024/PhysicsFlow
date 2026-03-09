"""
PhysicsFlow RAG — Query Processor.

Transforms a raw user question into multiple retrieval-optimised queries:
  1. Original query (always included)
  2. Sub-query decomposition (complex → simpler atomic queries)
  3. HyDE — Hypothetical Document Embedding (generate a fake answer, embed it)
  4. Keyword extraction for BM25 boosting

Uses the active Ollama model when available; falls back to rule-based
expansions when the LLM is offline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

try:
    import ollama
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ExpandedQuery:
    """All query variants generated from the original user question."""
    original:       str
    sub_queries:    list[str]    = field(default_factory=list)
    hyde_document:  Optional[str] = None
    keywords:       list[str]    = field(default_factory=list)

    @property
    def all_queries(self) -> list[str]:
        """All text strings to retrieve for (original + sub-queries + HyDE)."""
        queries = [self.original] + self.sub_queries
        if self.hyde_document:
            queries.append(self.hyde_document)
        return queries


# ── Reservoir-domain query classifier ─────────────────────────────────────────

_QUANTITATIVE_PATTERNS = re.compile(
    r'\b(how much|how many|what is the|give me|show me|plot|compare|'
    r'rate|pressure|mismatch|rmse|p10|p50|p90|eur|bhp|wopr|wwpr|wgpr)\b',
    re.IGNORECASE,
)
_WELL_PATTERN = re.compile(r'\b[A-Z]-\d+[A-Z]{0,2}\b')
_CONCEPTUAL_PATTERNS = re.compile(
    r'\b(explain|what is|how does|why|describe|meaning|definition|'
    r'difference between|compare|overview)\b',
    re.IGNORECASE,
)


def classify_query(query: str) -> str:
    """Return 'quantitative', 'conceptual', or 'general'."""
    if _QUANTITATIVE_PATTERNS.search(query):
        return "quantitative"
    if _CONCEPTUAL_PATTERNS.search(query):
        return "conceptual"
    return "general"


# ── Rule-based keyword extractor ──────────────────────────────────────────────

_STOP_WORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
    "has", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "not", "no", "in", "on", "at", "to", "for",
    "of", "with", "about", "and", "or", "but", "if", "then", "so", "as",
    "by", "from", "up", "down", "into", "this", "that", "these", "those",
    "it", "its", "i", "we", "you", "he", "she", "they", "me", "my", "our",
    "how", "what", "when", "where", "which", "who", "why",
])


def extract_keywords(query: str) -> list[str]:
    """Extract meaningful tokens from query for BM25 boosting."""
    # Preserve well names first
    wells = _WELL_PATTERN.findall(query)
    # Simple word extraction
    words = re.findall(r'\b[a-zA-Z0-9_\-]{2,}\b', query.lower())
    keywords = [w for w in words if w not in _STOP_WORDS]
    return list(dict.fromkeys(keywords + [w.lower() for w in wells]))


# ── LLM-based expansion ────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """You are a reservoir engineering assistant. Break the following
question into 2-3 simpler, specific sub-questions that together cover the full answer.
Return ONLY a JSON array of strings — no other text.

Question: {query}

Example output: ["What is the permeability of well B-2H?", "What is the water cut trend?"]
JSON array:"""

_HYDE_PROMPT = """You are a senior reservoir engineer. Write a short, factual paragraph
(50-80 words) that directly answers the following question, as if taken from a technical
report. Use realistic reservoir engineering terminology and plausible values.
Write ONLY the paragraph — no preamble, no labels.

Question: {query}

Answer paragraph:"""


class QueryProcessor:
    """
    Transforms a raw user query into multiple retrieval-optimised variants.

    Strategy:
    - Always: extract keywords + original query
    - When LLM available: generate sub-queries + HyDE document
    - Query type classification guides retrieval strategy (quantitative → tool first)

    Usage:
        qp = QueryProcessor(model="phi3:mini")
        expanded = qp.expand("Why is well B-2H producing so much water?")
        for q in expanded.all_queries:
            results = retriever.search(q)
    """

    def __init__(
        self,
        model: str = "phi3:mini",
        use_hyde: bool = True,
        use_decompose: bool = True,
        timeout_s: float = 8.0,
    ):
        self.model        = model
        self.use_hyde     = use_hyde
        self.use_decompose = use_decompose
        self.timeout_s    = timeout_s

    def expand(self, query: str) -> ExpandedQuery:
        """
        Expand a query into multiple retrieval variants.
        Falls back gracefully if Ollama is unavailable.
        """
        keywords = extract_keywords(query)
        query_type = classify_query(query)
        log.debug("Query type: %s | keywords: %s", query_type, keywords[:5])

        sub_queries: list[str] = []
        hyde_doc: Optional[str] = None

        if _HAS_OLLAMA:
            # Sub-query decomposition — only for complex / conceptual queries
            if self.use_decompose and len(query.split()) > 8:
                sub_queries = self._decompose(query)

            # HyDE — generate a hypothetical answer to embed densely
            if self.use_hyde and query_type != "quantitative":
                hyde_doc = self._generate_hyde(query)
        else:
            # Rule-based fallback: simple keyword-driven sub-queries
            sub_queries = self._rule_based_decompose(query, keywords)

        return ExpandedQuery(
            original=query,
            sub_queries=sub_queries,
            hyde_document=hyde_doc,
            keywords=keywords,
        )

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _decompose(self, query: str) -> list[str]:
        try:
            prompt = _DECOMPOSE_PROMPT.format(query=query)
            resp = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 200},
            )
            raw = resp.response.strip()
            # Extract JSON array
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                import json
                candidates = json.loads(match.group())
                return [str(c).strip() for c in candidates if c != query][:3]
        except Exception as e:
            log.debug("Sub-query decomposition failed: %s", e)
        return []

    def _generate_hyde(self, query: str) -> Optional[str]:
        try:
            prompt = _HYDE_PROMPT.format(query=query)
            resp = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.7, "num_predict": 150},
            )
            doc = resp.response.strip()
            if len(doc.split()) >= 20:
                return doc
        except Exception as e:
            log.debug("HyDE generation failed: %s", e)
        return None

    # ── Rule-based fallbacks ──────────────────────────────────────────────────

    @staticmethod
    def _rule_based_decompose(query: str, keywords: list[str]) -> list[str]:
        """Generate simple variants without LLM."""
        sub: list[str] = []

        # Well-name focused sub-query
        wells = _WELL_PATTERN.findall(query)
        for w in wells[:2]:
            sub.append(f"{w} production performance history")

        # Keyword combination (top 3 keywords as a focused query)
        if len(keywords) >= 3:
            sub.append(" ".join(keywords[:3]))

        return sub
