"""
Voicelab TTS plugin for LiveKit Agents.
Provides integration with Vogent's VoiceLab API for high-quality text-to-speech.
"""
from __future__ import annotations

from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__
from .plugin import VoiceLabPlugin

__all__ = ["TTS", "ChunkedStream", "SynthesizeStream", "__version__", "VoiceLabPlugin"]
