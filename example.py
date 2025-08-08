"""
Example usage of the VoiceLab TTS plugin for LiveKit Agents.
"""
import asyncio
import os
import wave
from typing import List

from livekit.agents import tts
from livekit.agents.voice.io import TimedString

from voicelab import TTS


async def basic_example():
    """Basic example of using VoiceLab TTS."""
    # Initialize the TTS engine
    tts_engine = TTS(
        api_key=os.environ.get("VOGENT_API_KEY"),
        voice_id="default",
        sample_rate=24000,
        word_timestamps=True
    )

    # Text to synthesize
    text = "Hello, world! This is VoiceLab TTS, a high-quality text-to-speech plugin for LiveKit Agents."

    # Create a collector to store the audio data
    collector = tts.AudioCollector()

    # Synthesize the text
    stream = tts_engine.synthesize(text)
    await stream.run(collector)

    # Save the audio to a WAV file
    save_to_wav(collector.audio_data, "basic_example.wav", collector.sample_rate)

    # Print the word timestamps
    print("Word timestamps:")
    for ts in collector.timed_transcript:
        print(f"  {ts.text.strip()}: {ts.start_time:.3f}s - {ts.end_time:.3f}s")

    # Clean up
    await tts_engine.aclose()


async def streaming_example():
    """Example of streaming text to speech incrementally."""
    # Initialize the TTS engine
    tts_engine = TTS(
        api_key=os.environ.get("VOGENT_API_KEY"),
        voice_id="default",
        sample_rate=24000,
        word_timestamps=True
    )

    # Create a collector to store the audio data
    collector = tts.AudioCollector()

    # Create a streaming synthesis session
    stream = tts_engine.stream()
    
    # Start the stream in the background
    stream_task = asyncio.create_task(stream.run(collector))
    
    # Add text incrementally
    sentences = [
        "This is a streaming example. ",
        "Text is sent to the TTS engine incrementally, ",
        "allowing for real-time speech synthesis. ",
        "This is particularly useful for conversational agents ",
        "that need to generate speech on the fly."
    ]
    
    for sentence in sentences:
        await stream.push_text(sentence)
        # Simulate some processing time
        await asyncio.sleep(0.5)
    
    # Close the stream
    await stream.aclose()
    
    # Wait for the stream to finish
    await stream_task
    
    # Save the audio to a WAV file
    save_to_wav(collector.audio_data, "streaming_example.wav", collector.sample_rate)
    
    # Clean up
    await tts_engine.aclose()


async def voice_customization_example():
    """Example of customizing the voice."""
    # Initialize the TTS engine with voice options
    tts_engine = TTS(
        api_key=os.environ.get("VOGENT_API_KEY"),
        voice_id="default",
        voice_options=[
            {"optionId": "speed", "value": "1.2"},
            {"optionId": "pitch", "value": "0.9"}
        ]
    )

    # Create a collector to store the audio data
    collector = tts.AudioCollector()

    # Synthesize text with the customized voice
    text = "This is an example of voice customization. The speed and pitch have been adjusted."
    stream = tts_engine.synthesize(text)
    await stream.run(collector)

    # Save the audio to a WAV file
    save_to_wav(collector.audio_data, "voice_customization.wav", collector.sample_rate)

    # Update the voice options
    tts_engine.update_options(
        voice_id="another_voice",
        voice_options=[
            {"optionId": "speed", "value": "0.8"},
            {"optionId": "pitch", "value": "1.1"}
        ]
    )

    # Create a new collector
    collector2 = tts.AudioCollector()

    # Synthesize text with the updated voice
    text2 = "Now the voice has been changed. The speed is slower and the pitch is higher."
    stream2 = tts_engine.synthesize(text2)
    await stream2.run(collector2)

    # Save the audio to a WAV file
    save_to_wav(collector2.audio_data, "voice_customization2.wav", collector2.sample_rate)

    # Clean up
    await tts_engine.aclose()


def save_to_wav(audio_data: List[bytes], filename: str, sample_rate: int):
    """Save audio data to a WAV file."""
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for chunk in audio_data:
            wav_file.writeframes(chunk)
    print(f"Audio saved to {filename}")


async def main():
    """Run all examples."""
    print("Running basic example...")
    await basic_example()
    
    print("\nRunning streaming example...")
    await streaming_example()
    
    print("\nRunning voice customization example...")
    await voice_customization_example()


if __name__ == "__main__":
    # Set your API key in the environment
    # os.environ["VOGENT_API_KEY"] = "your-api-key-here"
    
    # Run the examples
    asyncio.run(main())
