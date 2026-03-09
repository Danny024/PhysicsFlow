"""
PhysicsFlow RAG — Document Processor.

Converts raw sources (PDF, LAS, CSV, pfproj JSON, plain text, audit log)
into overlapping text chunks with rich metadata for indexing.

Chunking strategy:
  - Sentence-aware sliding window (chunk_size tokens, overlap_size tokens)
  - Tables and structured data kept together as atomic chunks
  - Metadata: source_path, source_type, page, section, domain_tags, hash
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import fitz                              # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

try:
    import lasio
    _HAS_LASIO = True
except ImportError:
    _HAS_LASIO = False


# ── Reservoir-domain stop-words (kept in BM25 but down-weighted) ─────────────

DOMAIN_KEYWORDS = frozenset([
    "permeability", "porosity", "pressure", "saturation", "reservoir",
    "well", "producer", "injector", "history", "matching", "simulation",
    "ensemble", "kalman", "pino", "areki", "fno", "surrogate", "training",
    "mismatch", "convergence", "epoch", "loss", "forecast", "eur",
    "p10", "p50", "p90", "wopr", "wwpr", "wgpr", "bhp", "wct",
    "norne", "eclipse", "compdat", "pvt", "bo", "bg", "rs", "viscosity",
    "transmissibility", "fault", "layer", "grid", "darcy", "peacemannn",
])


@dataclass
class DocumentChunk:
    """A single retrievable text chunk with metadata."""
    chunk_id:   str
    text:       str
    source_path: str
    source_type: str          # "pdf" | "las" | "csv" | "pfproj" | "text" | "audit" | "chat"
    page:        int  = 0
    section:     str  = ""
    domain_tags: list[str] = field(default_factory=list)
    metadata:    dict = field(default_factory=dict)
    score:       float = 0.0  # filled in by retriever

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]

    def citation(self) -> str:
        parts = [Path(self.source_path).name]
        if self.page > 0:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(self.section)
        return " › ".join(parts)


# ── Chunker ───────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Sentence-aware sliding window chunker.

    Splits text into sentences, then groups them into chunks of
    ~chunk_words words with overlap_words overlap.
    """

    # Simple sentence splitter that handles abbreviations
    _SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    def __init__(self, chunk_words: int = 180, overlap_words: int = 40):
        self.chunk_words   = chunk_words
        self.overlap_words = overlap_words

    def chunk(self, text: str) -> list[str]:
        text = self._clean(text)
        if not text:
            return []

        sentences = self._SENT_RE.split(text)
        chunks: list[str] = []
        current_words: list[str] = []
        current_len = 0

        for sent in sentences:
            words = sent.split()
            if not words:
                continue

            current_words.extend(words)
            current_len += len(words)

            if current_len >= self.chunk_words:
                chunks.append(" ".join(current_words))
                # Keep overlap
                overlap = current_words[-self.overlap_words:] if self.overlap_words else []
                current_words = list(overlap)
                current_len = len(current_words)

        if current_words:
            chunks.append(" ".join(current_words))

        return [c for c in chunks if len(c.split()) >= 10]  # drop tiny chunks

    @staticmethod
    def _clean(text: str) -> str:
        # Normalise whitespace, remove null bytes
        text = text.replace('\x00', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# ── Source parsers ────────────────────────────────────────────────────────────

class DocumentProcessor:
    """
    Auto-detects source type and routes to the correct parser.
    Returns a list of DocumentChunk objects ready for indexing.
    """

    def __init__(self, chunk_words: int = 180, overlap_words: int = 40):
        self.chunker = TextChunker(chunk_words, overlap_words)

    def process_file(self, path: str | Path) -> list[DocumentChunk]:
        p = Path(path)
        ext = p.suffix.lower()
        try:
            if ext == ".pdf":
                return self._process_pdf(p)
            elif ext == ".las":
                return self._process_las(p)
            elif ext in (".csv", ".tsv"):
                return self._process_csv(p)
            elif ext in (".pfproj",):
                return self._process_pfproj(p)
            elif ext in (".txt", ".md", ".rst"):
                return self._process_text(p)
            elif ext == ".json":
                return self._process_json(p)
            else:
                return []
        except Exception:
            return []

    def process_text(
        self,
        text: str,
        source_path: str,
        source_type: str = "text",
        metadata: Optional[dict] = None,
    ) -> list[DocumentChunk]:
        """Process an arbitrary text string into chunks."""
        chunks = self.chunker.chunk(text)
        return [
            self._make_chunk(c, source_path, source_type,
                             page=0, section="", meta=metadata or {})
            for c in chunks
        ]

    # ── PDF ───────────────────────────────────────────────────────────────────

    def _process_pdf(self, path: Path) -> list[DocumentChunk]:
        if not _HAS_FITZ:
            return self.process_text(
                f"[PDF: {path.name} — install PyMuPDF to extract text]",
                str(path), "pdf",
            )

        result: list[DocumentChunk] = []
        try:
            doc = fitz.open(str(path))
        except Exception:
            return []

        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            if not page_text.strip():
                continue

            # Detect tables (heuristic: many tab/pipe characters)
            if page_text.count('\t') > 20 or page_text.count('|') > 10:
                # Keep table pages as one atomic chunk
                result.append(self._make_chunk(
                    page_text[:2000], str(path), "pdf",
                    page=page_num, section="table",
                    meta={"has_table": True},
                ))
            else:
                for chunk in self.chunker.chunk(page_text):
                    result.append(self._make_chunk(
                        chunk, str(path), "pdf",
                        page=page_num, section="",
                    ))

        doc.close()
        return result

    # ── LAS well log ──────────────────────────────────────────────────────────

    def _process_las(self, path: Path) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        if _HAS_LASIO:
            try:
                las = lasio.read(str(path))
                # Header metadata as text
                header_lines = []
                for section in las.sections.values():
                    for item in section.values() if hasattr(section, 'values') else []:
                        header_lines.append(
                            f"{item.mnemonic}: {item.value} ({item.unit}) — {item.descr}"
                        )
                header_text = "\n".join(header_lines)

                # Column descriptions
                curve_desc = "\n".join(
                    f"{c.mnemonic} [{c.unit}]: {c.descr}"
                    for c in las.curves
                )

                # Zone summary (depth range + key curve stats)
                stats_lines = [f"Depth range: {las.index.min():.1f}–{las.index.max():.1f} {las.index_unit}"]
                for c in las.curves:
                    try:
                        vals = las[c.mnemonic]
                        stats_lines.append(
                            f"{c.mnemonic}: mean={vals.mean():.3f}, "
                            f"min={vals.min():.3f}, max={vals.max():.3f} {c.unit}"
                        )
                    except Exception:
                        pass

                full_text = f"LAS File: {path.name}\n\n" \
                            f"HEADER:\n{header_text}\n\n" \
                            f"CURVES:\n{curve_desc}\n\n" \
                            f"STATISTICS:\n" + "\n".join(stats_lines)

                for chunk in self.chunker.chunk(full_text):
                    chunks.append(self._make_chunk(
                        chunk, str(path), "las",
                        section="header+stats",
                        meta={"well": las.well.WELL.value if hasattr(las.well, 'WELL') else path.stem},
                    ))
            except Exception:
                pass

        if not chunks:
            # Fallback: read raw text
            try:
                raw = path.read_text(encoding='utf-8', errors='replace')[:4000]
                for chunk in self.chunker.chunk(raw):
                    chunks.append(self._make_chunk(chunk, str(path), "las"))
            except Exception:
                pass

        return chunks

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _process_csv(self, path: Path) -> list[DocumentChunk]:
        try:
            raw = path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            return []

        lines = raw.splitlines()
        if not lines:
            return []

        header = lines[0]
        # Summarise as text: header + first 20 rows + stats hint
        summary = f"CSV File: {path.name}\nColumns: {header}\nRows: {len(lines)-1}\n"
        # Include first 20 data rows verbatim
        summary += "\n".join(lines[1:21])
        if len(lines) > 21:
            summary += f"\n... ({len(lines)-21} more rows)"

        return [self._make_chunk(summary, str(path), "csv",
                                  meta={"n_rows": len(lines)-1, "columns": header})]

    # ── .pfproj project file ──────────────────────────────────────────────────

    def _process_pfproj(self, path: Path) -> list[DocumentChunk]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return []

        chunks: list[DocumentChunk] = []

        # Project overview
        overview = (
            f"Project: {data.get('project_name', path.stem)}\n"
            f"Version: {data.get('version', '?')}\n"
            f"Grid: {data.get('grid', {}).get('nx', '?')} × "
            f"{data.get('grid', {}).get('ny', '?')} × "
            f"{data.get('grid', {}).get('nz', '?')}\n"
            f"Wells: {len(data.get('wells', []))}\n"
            f"Notes: {data.get('notes', '')}\n"
            f"Eclipse deck: {data.get('eclipse_deck_path', 'None')}"
        )
        chunks.append(self._make_chunk(overview, str(path), "pfproj",
                                        section="overview"))

        # Wells summary
        wells = data.get("wells", [])
        if wells:
            well_text = f"Well list ({len(wells)} wells):\n"
            for w in wells:
                well_text += (
                    f"  {w.get('name', '?')} | type={w.get('type', '?')} | "
                    f"perfs: K={w.get('k_top', '?')}-{w.get('k_bot', '?')}\n"
                )
            chunks.append(self._make_chunk(well_text, str(path), "pfproj",
                                            section="wells"))

        # Notes (verbatim — high information density)
        notes = data.get("notes", "")
        if notes and len(notes) > 20:
            for chunk in self.chunker.chunk(notes):
                chunks.append(self._make_chunk(chunk, str(path), "pfproj",
                                                section="notes"))

        # HM results summary
        hm = data.get("hm_results", {})
        if hm:
            hm_text = (
                f"History Matching Results:\n"
                f"  Converged: {hm.get('converged', False)}\n"
                f"  Best mismatch: {hm.get('best_mismatch', 'N/A')}\n"
                f"  Iterations: {hm.get('n_iterations', 0)}\n"
                f"  P10 EUR: {hm.get('eur_p10', 'N/A')} MMstb\n"
                f"  P50 EUR: {hm.get('eur_p50', 'N/A')} MMstb\n"
                f"  P90 EUR: {hm.get('eur_p90', 'N/A')} MMstb\n"
            )
            chunks.append(self._make_chunk(hm_text, str(path), "pfproj",
                                            section="hm_results"))

        return chunks

    # ── Plain text / Markdown ─────────────────────────────────────────────────

    def _process_text(self, path: Path) -> list[DocumentChunk]:
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            return []

        # Detect Markdown sections
        sections = re.split(r'\n#{1,3}\s+', text)
        chunks: list[DocumentChunk] = []
        for i, sec in enumerate(sections):
            for chunk in self.chunker.chunk(sec):
                chunks.append(self._make_chunk(
                    chunk, str(path), "text",
                    section=f"section_{i}",
                ))
        return chunks

    # ── Generic JSON ──────────────────────────────────────────────────────────

    def _process_json(self, path: Path) -> list[DocumentChunk]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            text = json.dumps(data, indent=2)[:6000]
        except Exception:
            return []
        return [self._make_chunk(text, str(path), "json")]

    # ── Audit log entries ─────────────────────────────────────────────────────

    def process_audit_entries(self, entries: list[dict]) -> list[DocumentChunk]:
        """Convert SQLite audit_log rows to searchable chunks."""
        if not entries:
            return []

        # Group into batches of 20 for chunking
        chunks: list[DocumentChunk] = []
        batch: list[str] = []

        for e in entries:
            line = (
                f"[{e.get('timestamp', '?')}] "
                f"{e.get('event_type', '?')}: "
                f"{e.get('description', '')}"
            )
            if e.get('project_name'):
                line += f" (project: {e['project_name']})"
            batch.append(line)

            if len(batch) >= 20:
                text = "\n".join(batch)
                for chunk in self.chunker.chunk(text):
                    chunks.append(self._make_chunk(
                        chunk, "audit_log", "audit",
                        section="events",
                    ))
                batch = []

        if batch:
            text = "\n".join(batch)
            chunks.append(self._make_chunk(text, "audit_log", "audit",
                                            section="events"))
        return chunks

    # ── Factory helper ────────────────────────────────────────────────────────

    def _make_chunk(
        self,
        text: str,
        source_path: str,
        source_type: str,
        page: int = 0,
        section: str = "",
        meta: Optional[dict] = None,
    ) -> DocumentChunk:
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        chunk_id = f"{Path(source_path).stem}_{source_type}_{content_hash}"

        # Auto-tag with reservoir domain keywords found in text
        text_lower = text.lower()
        tags = [kw for kw in DOMAIN_KEYWORDS if kw in text_lower]

        return DocumentChunk(
            chunk_id    = chunk_id,
            text        = text,
            source_path = source_path,
            source_type = source_type,
            page        = page,
            section     = section,
            domain_tags = tags,
            metadata    = meta or {},
        )
