"""
PhysicsFlow RAG — Context Builder.

Assembles the final LLM context from:
  - Reranked document chunks (with citations)
  - Live tool-call data (simulation state, well results, etc.)
  - Conversation history summary

Produces a structured XML-tagged context block injected into the
system prompt, respecting a token budget.

Output format (injected into system prompt):
    <retrieved_context>
      <document index="1" source="Norne_HM_Report.pdf" page="12" score="0.92">
        ... chunk text ...
      </document>
      ...
    </retrieved_context>
    <citations>
      [1] Norne_HM_Report.pdf p.12  [2] well_data.las ...
    </citations>
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .document_processor import DocumentChunk

log = logging.getLogger(__name__)

# Approximate token → word ratio (conservative for field content)
_WORDS_PER_TOKEN = 0.75
# Max tokens reserved for RAG context in the LLM prompt
_DEFAULT_TOKEN_BUDGET = 1800


@dataclass
class RetrievedContext:
    """Structured context ready for injection into the system prompt."""
    context_block: str       # XML-tagged context text
    citations:     str       # formatted citation list
    chunk_count:   int       # number of chunks included
    sources:       list[str] # unique source paths


class ContextBuilder:
    """
    Converts a ranked list of DocumentChunks into a formatted LLM context.

    Features:
    - Token budget enforcement (truncates chunks to fit)
    - Deduplication by chunk_id
    - Citation numbering [1], [2], …
    - XML-structured output for clear context boundaries
    - Optional tool-data injection

    Usage:
        builder = ContextBuilder(token_budget=1800)
        ctx = builder.build(reranked_chunks)
        system_prompt += ctx.context_block
    """

    def __init__(self, token_budget: int = _DEFAULT_TOKEN_BUDGET):
        self.token_budget = token_budget

    def build(
        self,
        chunks: list[tuple[DocumentChunk, float]],
        tool_data: Optional[dict] = None,
    ) -> RetrievedContext:
        """
        Build context from reranked chunks and optional live tool data.

        Args:
            chunks:    (chunk, score) pairs, sorted by relevance.
            tool_data: Optional dict of live simulation results to inject.

        Returns:
            RetrievedContext with formatted context_block and citations.
        """
        if not chunks:
            return RetrievedContext(
                context_block="", citations="", chunk_count=0, sources=[]
            )

        word_budget = int(self.token_budget / _WORDS_PER_TOKEN)

        # Deduplicate by chunk_id
        seen_ids: set[str] = set()
        deduped: list[tuple[DocumentChunk, float]] = []
        for chunk, score in chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                deduped.append((chunk, score))

        # Allocate word budget across chunks
        doc_blocks: list[str] = []
        citations:  list[str] = []
        sources:    list[str] = []
        words_used = 0

        for i, (chunk, score) in enumerate(deduped, start=1):
            words = chunk.text.split()
            remaining = word_budget - words_used

            if remaining <= 20:
                break   # budget exhausted

            # Truncate chunk to remaining budget
            if len(words) > remaining:
                text = " ".join(words[:remaining]) + " …"
            else:
                text = chunk.text

            words_used += min(len(words), remaining)

            citation = chunk.citation()
            doc_blocks.append(
                f'  <document index="{i}" source="{citation}" score="{score:.3f}">\n'
                f'    {text}\n'
                f'  </document>'
            )
            citations.append(f"[{i}] {citation}")
            if chunk.source_path not in sources:
                sources.append(chunk.source_path)

        # Optionally prepend live tool data
        tool_block = ""
        if tool_data:
            tool_lines = []
            for key, val in tool_data.items():
                if isinstance(val, (str, int, float, bool)):
                    tool_lines.append(f"    <{key}>{val}</{key}>")
            if tool_lines:
                tool_block = (
                    "  <live_data>\n"
                    + "\n".join(tool_lines)
                    + "\n  </live_data>\n"
                )

        context_block = (
            "<retrieved_context>\n"
            + tool_block
            + "\n".join(doc_blocks)
            + "\n</retrieved_context>"
        )
        citation_text = "\n".join(citations)

        log.debug(
            "Context built: %d chunks, %d words, %d sources",
            len(doc_blocks), words_used, len(sources),
        )

        return RetrievedContext(
            context_block=context_block,
            citations=citation_text,
            chunk_count=len(doc_blocks),
            sources=sources,
        )

    def format_for_prompt(self, ctx: RetrievedContext) -> str:
        """Return the full text to append to the system prompt."""
        if not ctx.context_block:
            return ""
        parts = ["## Relevant Knowledge Base Context", ctx.context_block]
        if ctx.citations:
            parts += ["## Sources", ctx.citations]
        parts.append(
            "\nUse the above retrieved context to ground your answer. "
            "Cite sources as [1], [2], etc. when referencing specific documents."
        )
        return "\n\n".join(parts)
