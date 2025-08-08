# VoiceLab TTS Plugin for LiveKit Agents

A high-quality text-to-speech plugin for LiveKit Agents that integrates with Vogent's VoiceLab API.

## Overview

VoiceLab TTS is a plugin for LiveKit Agents that provides high-quality text-to-speech capabilities through Vogent's VoiceLab API. This plugin enables your LiveKit applications to generate natural-sounding speech from text input, with support for streaming, word timestamps, and various voice customization options.

## Features

- High-quality text-to-speech synthesis
- Streaming support for real-time audio generation
- Word-level timestamps for precise audio-text alignment
- Customizable voice options
- Multiple sample rate support (8kHz to 48kHz)
- Sentence tokenization and pacing for natural speech flow

## Installation

```bash
pip install livekit-voicelab
```

## Requirements

- Python 3.7+
- An API key from Vogent's VoiceLab service
- LiveKit Agents SDK

## Configuration

Set your Vogent API key as an environment variable:

```bash
export VOGENT_API_KEY="your-api-key-here"
```

Alternatively, you can provide the API key directly when initializing the TTS class.

## Usage

### Basic Usage

```python
from voicelab import TTS

# Initialize the TTS engine
tts = TTS(
    api_key="your-api-key-here",  # Optional if set as environment variable
    voice_id="default",           # Optional, defaults to DEFAULT_VOICE_ID
    sample_rate=24000,            # Optional, defaults to 24000
    word_timestamps=True          # Optional, defaults to True
)

# Synthesize text to speech in one go
audio_stream = tts.synthesize("Hello, world! This is VoiceLab TTS.")

# Process the audio stream
# ...
```

### Streaming Usage

```python
from voicelab import TTS

# Initialize the TTS engine
tts = TTS(api_key="your-api-key-here")

# Create a streaming synthesis session
stream = tts.stream()

# Add text incrementally
await stream.push_text("This is a streaming ")
await stream.push_text("text-to-speech example.")

# Flush the current sentence
await stream.flush()

# Close the stream when done
await stream.aclose()
```

### Voice Customization

```python
from voicelab import TTS

# Initialize with voice options
tts = TTS(
    voice_id="custom_voice",
    voice_options=[
        {"optionId": "speed", "value": "1.2"},
        {"optionId": "pitch", "value": "0.9"}
    ]
)

# Update options later
tts.update_options(
    voice_id="another_voice",
    voice_options=[
        {"optionId": "speed", "value": "1.0"},
        {"optionId": "pitch", "value": "1.0"}
    ]
)
```

## Advanced Configuration

The TTS class accepts several parameters for advanced configuration:

- `api_key`: Your Vogent API key
- `voice_id`: ID of the voice to use
- `sample_rate`: Audio sample rate in Hz (8000, 16000, 22050, 24000, 44100, or 48000)
- `word_timestamps`: Whether to include word-level timestamps
- `voice_options`: List of voice customization options
- `http_session`: Custom aiohttp ClientSession
- `tokenizer`: Custom sentence tokenizer
- `text_pacing`: Controls the pacing of streamed text
- `base_url`: Custom API endpoint URL

## Error Handling

The plugin provides detailed error handling for various scenarios:

- `APIConnectionError`: Connection issues with the Vogent API
- `APITimeoutError`: Request timeout
- `APIStatusError`: HTTP status errors
- `APIError`: General API errors

## License

[Include license information here]

## Contributing

[Include contribution guidelines here]
