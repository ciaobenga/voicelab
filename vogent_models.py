"""
Models and constants for the Vogent VoiceLab API integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

# Default configuration values
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_VOICE_ID = "default"
DEFAULT_BASE_URL = "wss://api.vogent.ai/tts/stream"
VOGENT_API_KEY_ENV = "VOGENT_API_KEY"

class SampleRate(Enum):
    """Valid sample rates for Vogent TTS."""
    RATE_8K = 8000
    RATE_16K = 16000
    RATE_22K = 22050
    RATE_24K = 24000
    RATE_44K = 44100
    RATE_48K = 48000

@dataclass
class VoiceOption:
    """Voice customization option for Vogent TTS."""
    option_id: str
    value: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API requests."""
        return {
            "optionId": self.option_id,
            "value": self.value
        }
