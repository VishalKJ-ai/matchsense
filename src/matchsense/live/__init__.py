"""Live pipeline: replay a match as a feed and run the analytics on it as it happens."""

from .feed import Frame, replay

__all__ = ["Frame", "replay"]
