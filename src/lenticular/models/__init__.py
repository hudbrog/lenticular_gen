"""Model wrappers: depth estimation, segmentation, inpainting.

These modules import heavy ML libraries (torch, transformers, rembg) lazily
inside their functions, so the rest of the package stays importable without the
optional ``ml`` dependency group installed.
"""

from __future__ import annotations
