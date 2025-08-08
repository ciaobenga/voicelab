"""
Example of integrating the VoiceLab TTS plugin with LiveKit Agents.
"""
import asyncio
import os
from typing import List

from livekit.agents import Agent, AgentConfig
from livekit.agents.voice.io import TimedString
from livekit.plugins import load_plugin

# Import the plugin directly (alternative to loading via entry point)
from voicelab import VoiceLabPlugin


async def example_with_direct_import():
    """Example of using the VoiceLab plugin by direct import."""
    # Create an agent configuration
    config = AgentConfig(
        name="VoiceLab Demo Agent",
        tts=VoiceLabPlugin.get_components()["tts"](
            api_key=os.environ.get("VOGENT_API_KEY"),
            voice_id="default",
            sample_rate=24000,
            word_timestamps=True,
        ),
    )

    # Create an agent with the configuration
    agent = Agent(config)

    # Use the agent with VoiceLab TTS
    await agent.tts.synthesize("Hello, I'm using VoiceLab TTS through direct import!").run()

    # Clean up
    await agent.aclose()


async def example_with_plugin_loading():
    """Example of using the VoiceLab plugin via plugin loading mechanism."""
    # Load the plugin
    plugin = load_plugin("voicelab")
    
    # Get the TTS component from the plugin
    tts_class = plugin.get_components()["tts"]
    
    # Create an agent configuration with the TTS component
    config = AgentConfig(
        name="VoiceLab Plugin Demo Agent",
        tts=tts_class(
            api_key=os.environ.get("VOGENT_API_KEY"),
            voice_id="default",
            sample_rate=24000,
            word_timestamps=True,
        ),
    )

    # Create an agent with the configuration
    agent = Agent(config)

    # Use the agent with VoiceLab TTS
    await agent.tts.synthesize("Hello, I'm using VoiceLab TTS through plugin loading!").run()

    # Clean up
    await agent.aclose()


async def main():
    """Run the examples."""
    print("Running example with direct import...")
    await example_with_direct_import()
    
    print("\nRunning example with plugin loading...")
    await example_with_plugin_loading()


if __name__ == "__main__":
    # Set your API key in the environment
    # os.environ["VOGENT_API_KEY"] = "your-api-key-here"
    
    # Run the examples
    asyncio.run(main())
