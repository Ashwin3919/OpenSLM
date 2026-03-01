"""Abstract base class defining the pipeline lifecycle interface.

All concrete pipelines inherit from ``BasePipeline`` and implement its
abstract methods.  The ``execute()`` method runs the full lifecycle in
the canonical order: configure → validate → run → save_results.
"""

from abc import ABC, abstractmethod

from src.models.config import AppConfig


class BasePipeline(ABC):
    """Base interface for all pipeline stages.

    Lifecycle contract:
        1. ``configure()`` — build all components from config (no side effects
           outside of ``self``).
        2. ``validate()`` — check prerequisites: required files exist, GPU is
           available, etc.  Raise early with a clear message on failure.
        3. ``run()`` — execute the core workflow.
        4. ``save_results()`` — persist outputs (metrics, artefacts).

    Subclasses that produce no persistent artefacts may leave
    ``save_results()`` as a no-op.

    Args:
        config: The fully-loaded and validated ``AppConfig`` for this run.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @abstractmethod
    def configure(self) -> None:
        """Load config values and construct all pipeline components."""

    @abstractmethod
    def validate(self) -> None:
        """Assert that all prerequisites are met before ``run()`` is called."""

    @abstractmethod
    def run(self) -> None:
        """Execute the pipeline's core workflow."""

    def save_results(self) -> None:
        """Persist outputs to disk.  Override in pipelines that produce artefacts."""

    def execute(self) -> None:
        """Run the full pipeline lifecycle in canonical order.

        Calls configure → validate → run → save_results in sequence.
        Exceptions from any stage propagate to the caller unmodified.
        """
        self.configure()
        self.validate()
        self.run()
        self.save_results()
