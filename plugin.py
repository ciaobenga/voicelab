# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VoiceLab plugin configuration for LiveKit.
"""
from __future__ import annotations

from typing import Dict, Any

from livekit.plugins import Plugin

from .version import __version__
from .tts import TTS


class VoiceLabPlugin(Plugin):
    """VoiceLab plugin for LiveKit Agents."""

    @classmethod
    def get_name(cls) -> str:
        """Get the name of the plugin."""
        return "voicelab"

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the plugin."""
        return __version__

    @classmethod
    def get_description(cls) -> str:
        """Get the description of the plugin."""
        return "VoiceLab TTS plugin for LiveKit Agents"

    @classmethod
    def get_components(cls) -> Dict[str, Any]:
        """Get the components provided by the plugin."""
        return {
            "tts": TTS,
        }
