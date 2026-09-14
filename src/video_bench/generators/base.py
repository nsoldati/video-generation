"""Common generator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    output_path: Path
    generation_parameters: dict[str, Any] = field(default_factory=dict)
    actual_num_frames: int | None = None


class VideoGenerator(ABC):
    """Interface implemented by every local text-to-video backend."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        seed: int,
        output_path: str | Path,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate one video and save it at ``output_path``."""
