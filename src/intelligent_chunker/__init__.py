"""Two-pass, model-native intelligent chunker for unstructured PDFs.

Pass 1 (analyze): read the whole document with Claude Haiku and build a global
map -- section outline, document-wide metadata, glossary, cross-references.

Pass 2 (chunker): re-read each section with the global map as context and emit
coherent, boundary-respecting chunks enriched with that context.
"""

from .config import ChunkerConfig
from .models import Chunk, DocumentProfile, Section

__all__ = ["Chunk", "DocumentProfile", "Section", "ChunkerConfig"]
__version__ = "0.1.0"
