"""Core data structures shared across the pipeline.

These are plain dataclasses with explicit ``to_dict`` / ``from_dict`` so they
round-trip cleanly to JSON (the global map) and JSONL (the chunks), and so the
structured-output schemas the model fills in stay in lock-step with the code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Section:
    """One contiguous section of the document, located by page range."""

    title: str
    section_type: str
    summary: str
    page_start: int
    page_end: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Section":
        return cls(
            title=d["title"],
            section_type=d.get("section_type", "unknown"),
            summary=d.get("summary", ""),
            page_start=int(d["page_start"]),
            page_end=int(d["page_end"]),
        )


@dataclass
class GlossaryTerm:
    term: str
    definition: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GlossaryTerm":
        return cls(term=d["term"], definition=d.get("definition", ""))


@dataclass
class DocumentProfile:
    """The global map produced by Pass 1.

    Holds everything we learn about the document as a whole, so Pass 2 can
    chunk each section with full context instead of reading blind.
    """

    source_file: str
    page_count: int
    doc_type: str = "unknown"
    title: str = ""
    plan_name: str = ""
    sponsor: str = ""
    effective_dates: List[str] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    glossary: List[GlossaryTerm] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "page_count": self.page_count,
            "doc_type": self.doc_type,
            "title": self.title,
            "plan_name": self.plan_name,
            "sponsor": self.sponsor,
            "effective_dates": list(self.effective_dates),
            "sections": [s.to_dict() for s in self.sections],
            "glossary": [g.to_dict() for g in self.glossary],
            "cross_references": list(self.cross_references),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DocumentProfile":
        return cls(
            source_file=d.get("source_file", ""),
            page_count=int(d.get("page_count", 0)),
            doc_type=d.get("doc_type", "unknown"),
            title=d.get("title", ""),
            plan_name=d.get("plan_name", ""),
            sponsor=d.get("sponsor", ""),
            effective_dates=list(d.get("effective_dates", [])),
            sections=[Section.from_dict(s) for s in d.get("sections", [])],
            glossary=[GlossaryTerm.from_dict(g) for g in d.get("glossary", [])],
            cross_references=list(d.get("cross_references", [])),
            notes=d.get("notes", ""),
        )


@dataclass
class Chunk:
    """One embedding-ready unit produced by Pass 2.

    ``text`` is the content to embed. Everything else is metadata carried
    alongside it for retrieval, filtering, and source attribution.
    """

    text: str
    source_file: str
    chunk_index: int
    section_title: str
    section_type: str
    section_summary: str
    page_start: int
    page_end: int
    keywords: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    token_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        return cls(
            text=d["text"],
            source_file=d.get("source_file", ""),
            chunk_index=int(d.get("chunk_index", 0)),
            section_title=d.get("section_title", ""),
            section_type=d.get("section_type", "unknown"),
            section_summary=d.get("section_summary", ""),
            page_start=int(d.get("page_start", 0)),
            page_end=int(d.get("page_end", 0)),
            keywords=list(d.get("keywords", [])),
            cross_references=list(d.get("cross_references", [])),
            token_count=d.get("token_count"),
        )
