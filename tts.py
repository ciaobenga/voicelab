"""
Voicelab TTS plugin for LiveKit Agents.
Provides integration with Vogent's VoiceLab API for high-quality text-to-speech.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
import weakref
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Union, cast

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString
from livekit.agents.log import logger

from vogent_models import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE_ID,
    DEFAULT_BASE_URL,
    VOGENT_API_KEY_ENV,
    VoiceOption,
    SampleRate,
)

@dataclass
class _TTSOptions:
    """Options for Vogent TTS."""
    voice_id: str
    sample_rate: int
    api_key: str
    base_url: str
    word_timestamps: bool
    voice_options: List[VoiceOption]

class TTS(tts.TTS):
    """
    Vogent TTS implementation for LiveKit Agents.
    """
    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_id: str = DEFAULT_VOICE_ID,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        word_timestamps: bool = True,
        voice_options: List[Dict[str, str]] = None,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        text_pacing: tts.SentenceStreamPacer | bool = False,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        """
        Create a new instance of Vogent TTS.

        Args:
            api_key (str, optional): The Vogent API key. If not provided, it will be read from the VOGENT_API_KEY environment variable.
            voice_id (str, optional): The ID of the voice to use. Defaults to DEFAULT_VOICE_ID.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            word_timestamps (bool, optional): Whether to add word timestamps to the output. Defaults to True.
            voice_options (List[Dict[str, str]], optional): Additional voice options to pass to the API.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
            tokenizer (tokenize.SentenceTokenizer, optional): The tokenizer to use.
            text_pacing (tts.SentenceStreamPacer | bool, optional): Stream pacer for the TTS. Set to True to use the default pacer, False to disable.
            base_url (str, optional): The base URL for the Vogent API. Defaults to DEFAULT_BASE_URL.
        """
        # Validate sample rate
        valid_sample_rates = [8000, 16000, 22050, 24000, 44100, 48000]
        if sample_rate not in valid_sample_rates:
            logger.warning(f"Invalid sample rate: {sample_rate}. Must be one of {valid_sample_rates}. Using default: {DEFAULT_SAMPLE_RATE}")
            sample_rate = DEFAULT_SAMPLE_RATE
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=word_timestamps,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )
        
        vogent_api_key = api_key or os.environ.get("VOGENT_API_KEY")
        if not vogent_api_key:
            raise ValueError("VOGENT_API_KEY must be set")

        # Convert voice_options to VoiceOption objects
        voice_option_objects = []
        if voice_options:
            for option in voice_options:
                if isinstance(option, dict) and "optionId" in option and "value" in option:
                    voice_option_objects.append(VoiceOption(option["optionId"], option["value"]))
        
        self._opts = _TTSOptions(
            voice_id=voice_id,
            sample_rate=sample_rate,
            api_key=vogent_api_key,
            base_url=base_url,
            word_timestamps=word_timestamps,
            voice_options=voice_option_objects,
        )
        
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        self._stream_pacer: tts.SentenceStreamPacer | None = None
        if text_pacing is True:
            self._stream_pacer = tts.SentenceStreamPacer()
        elif isinstance(text_pacing, tts.SentenceStreamPacer):
            self._stream_pacer = text_pacing
            
        logger.info(f"Initialized Vogent TTS with voice_id={voice_id}, sample_rate={sample_rate}")

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        """Connect to the Vogent websocket API."""
        session = self._ensure_session()
        
        # Add API key as query parameter
        url = f"{self._opts.base_url}?apiKey={self._opts.api_key}"
        logger.info(f"Connecting to Vogent API at {self._opts.base_url}")
        
        # Connect with authorization header
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}"
        }
        
        try:
            ws = await asyncio.wait_for(session.ws_connect(url, headers=headers), timeout)
            logger.info("Successfully connected to Vogent API websocket")
            return ws
        except Exception as e:
            logger.error(f"Error connecting to Vogent API: {e}")
            raise

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Close the websocket connection."""
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure we have an HTTP session."""
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def prewarm(self) -> None:
        """Prewarm the connection pool."""
        self._pool.prewarm()

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        voice_options: NotGivenOr[List[Dict[str, str]]] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        Args:
            voice_id (str, optional): The ID of the voice to use.
            voice_options (List[Dict[str, str]], optional): Additional voice options to pass to the API.
        """
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(voice_options):
            # Convert voice_options to VoiceOption objects
            voice_option_objects = []
            for option in voice_options:
                if isinstance(option, dict) and "optionId" in option and "value" in option:
                    voice_option_objects.append(VoiceOption(option["optionId"], option["value"]))
            self._opts.voice_options = voice_option_objects

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """Synthesize text to speech in one go."""
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Stream text to speech incrementally."""
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close all streams and the connection pool."""
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the Vogent API."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the chunked synthesis."""
        try:
            # For chunked synthesis, we'll use the websocket API but send the entire text at once
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                # Initialize the audio emitter
                request_id = str(uuid.uuid4())
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                )
                
                # Create the message to send
                # Ensure sample rate is one of the valid values
                valid_sample_rates = [8000, 16000, 22050, 24000, 44100, 48000]
                sample_rate = self._opts.sample_rate
                if sample_rate not in valid_sample_rates:
                    logger.warning(f"Invalid sample rate in chunked message: {sample_rate}. Forcing to 24000.")
                    sample_rate = 24000
                
                message = {
                    "generationId": request_id,
                    "voiceId": self._opts.voice_id,
                    "text": self._input_text,
                    "finalText": True,
                    "sampleRate": sample_rate,
                    "addWordTimestamps": self._opts.word_timestamps,
                }
                
                # Reduce logging to improve performance
                if len(self._input_text) > 50:
                    logger.info(f"Sending chunked message to Vogent API with text length: {len(self._input_text)}")
                else:
                    logger.debug(f"Sending chunked message to Vogent API: {message}")
                
                # Add voice options if provided
                if self._opts.voice_options:
                    message["voiceOptionValues"] = [option.to_dict() for option in self._opts.voice_options]
                
                # Send the message
                await ws.send_str(json.dumps(message))
                
                # Process the response
                current_segment_id = request_id
                output_emitter.start_segment(segment_id=current_segment_id)
                
                while True:
                    msg = await ws.receive()
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        break
                    
                    if msg.type != aiohttp.WSMsgType.BINARY and msg.type != aiohttp.WSMsgType.TEXT:
                        logger.warning(f"Unexpected Vogent message type: {msg.type}")
                        continue
                    
                    # Handle binary audio data
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        logger.debug(f"Received binary audio data of length {len(msg.data)}")
                        output_emitter.push(msg.data)
                        continue
                    
                    # Handle text messages (metadata, timestamps, etc.)
                    try:
                        data = json.loads(msg.data)
                        # Reduce logging to improve performance
                        if data.get("type") == "chunk" and data.get("audio"):
                            logger.debug("Received audio chunk in text message format")
                        else:
                            logger.debug(f"Received text message from Vogent API: {data}")
                        
                        # Check for errors
                        if data.get("error"):
                            raise APIError(f"Vogent returned error: {data['error']}")
                            
                        # Check for audio data in text message (base64 encoded)
                        if data.get("type") == "chunk" and data.get("audio"):
                            import base64
                            logger.debug("Received audio chunk in text message")
                            audio_data = base64.b64decode(data["audio"])
                            logger.debug(f"Decoded audio data of length {len(audio_data)}")
                            output_emitter.push(audio_data)
                        
                        # Handle word timestamps
                        if self._opts.word_timestamps and data.get("wordTimestamps"):
                            for word_data in data["wordTimestamps"]:
                                word = word_data.get("word", "")
                                start = word_data.get("startTime", 0)
                                end = word_data.get("endTime", 0)
                                output_emitter.push_timed_transcript(
                                    TimedString(text=word + " ", start_time=start, end_time=end)
                                )
                        
                        # Check if generation is complete
                        if data.get("done"):
                            break
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Vogent message: {msg.data}")
                
                # Finalize the output
                output_emitter.flush()
                
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Stream text to speech incrementally using the Vogent API."""
    
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._generation_id = str(uuid.uuid4())
        self._context_id = str(uuid.uuid4())

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the streaming synthesis."""
        request_id = self._generation_id
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()
        if self._tts._stream_pacer:
            sent_tokenizer_stream = self._tts._stream_pacer.wrap(
                sent_stream=sent_tokenizer_stream,
                audio_emitter=output_emitter,
            )

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Send sentences to the Vogent API as they become available."""
            first_message = True
            
            async for ev in sent_tokenizer_stream:
                # Create the message
                # Ensure sample rate is one of the valid values
                valid_sample_rates = [8000, 16000, 22050, 24000, 44100, 48000]
                sample_rate = self._opts.sample_rate
                if sample_rate not in valid_sample_rates:
                    logger.warning(f"Invalid sample rate in message: {sample_rate}. Forcing to 24000.")
                    sample_rate = 24000
                
                message = {
                    "generationId": self._generation_id,
                    "contextId": self._context_id,
                    "text": ev.token + " ",
                    "finalText": False,
                    "sampleRate": sample_rate,
                    "addWordTimestamps": self._opts.word_timestamps,
                }
                
                # Reduce logging to improve performance
                if len(ev.token) > 50:
                    logger.info(f"Sending message to Vogent API with text length: {len(ev.token)}")
                else:
                    logger.debug(f"Sending message to Vogent API: {message}")
                
                # Add voice ID only on the first message
                if first_message:
                    message["voiceId"] = self._opts.voice_id
                    
                    # Add voice options if provided
                    if self._opts.voice_options:
                        message["voiceOptionValues"] = [option.to_dict() for option in self._opts.voice_options]
                    
                    first_message = False
                
                # Send the message
                self._mark_started()
                await ws.send_str(json.dumps(message))

            # Send final empty message to indicate end of stream
            # Ensure sample rate is one of the valid values
            valid_sample_rates = [8000, 16000, 22050, 24000, 44100, 48000]
            sample_rate = self._opts.sample_rate
            if sample_rate not in valid_sample_rates:
                logger.warning(f"Invalid sample rate in final message: {sample_rate}. Forcing to 24000.")
                sample_rate = 24000
                
            final_message = {
                "generationId": self._generation_id,
                "contextId": self._context_id,
                "text": " ",
                "finalText": True,
                "sampleRate": sample_rate,
                "voiceId": self._opts.voice_id,  # Always include voice_id in final message
            }
            logger.debug(f"Sending final message to Vogent API: {final_message}")
            await ws.send_str(json.dumps(final_message))

        async def _input_task() -> None:
            """Process input text and send it to the tokenizer."""
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue

                sent_tokenizer_stream.push_text(data)

            sent_tokenizer_stream.end_input()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Receive and process messages from the Vogent API."""
            output_emitter.start_segment(segment_id=self._generation_id)
            
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Vogent connection closed unexpectedly", request_id=request_id
                    )

                # Handle binary audio data
                if msg.type == aiohttp.WSMsgType.BINARY:
                    logger.debug(f"Received binary audio data of length {len(msg.data)}")
                    output_emitter.push(msg.data)
                    continue
                
                # Handle text messages
                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning(f"Unexpected Vogent message type: {msg.type}")
                    continue
                
                try:
                    data = json.loads(msg.data)
                    # Reduce logging to improve performance
                    if data.get("type") == "chunk" and data.get("audio"):
                        logger.debug("Received audio chunk in text message format (stream)")
                    else:
                        logger.debug(f"Received text message from Vogent API in stream: {data}")
                    
                    # Check for errors
                    if data.get("error"):
                        raise APIError(f"Vogent returned error: {data['error']}")
                    
                    # Check for audio data in text message (base64 encoded)
                    if data.get("type") == "chunk" and data.get("audio"):
                        import base64
                        logger.debug("Received audio chunk in text message (stream)")
                        audio_data = base64.b64decode(data["audio"])
                        logger.debug(f"Decoded audio data of length {len(audio_data)}")
                        output_emitter.push(audio_data)
                    
                    # Handle word timestamps
                    if self._opts.word_timestamps and data.get("wordTimestamps"):
                        for word_data in data["wordTimestamps"]:
                            word = word_data.get("word", "")
                            start = word_data.get("startTime", 0)
                            end = word_data.get("endTime", 0)
                            output_emitter.push_timed_transcript(
                                TimedString(text=word + " ", start_time=start, end_time=end)
                            )
                    
                    # Check if generation is complete
                    if data.get("done") and sent_tokenizer_stream.closed:
                        output_emitter.end_input()
                        break
                        
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse Vogent message: {msg.data}")

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_sentence_stream_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    await sent_tokenizer_stream.aclose()
                    await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
