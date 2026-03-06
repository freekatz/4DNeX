"""Adapters that convert heterogeneous raw datasets into VideoClip instances."""

from core.datasets.adapters.base import VideoClip, VideoClipDescriptor
from core.datasets.adapters.fdnex import collect as collect_4dnex
from core.datasets.adapters.omniworld_game import collect as collect_omniworld_game
from core.datasets.adapters.omniworld_hoi4d import collect as collect_omniworld_hoi4d

__all__ = [
    "VideoClip",
    "VideoClipDescriptor",
    "collect_4dnex",
    "collect_omniworld_game",
    "collect_omniworld_hoi4d",
]
