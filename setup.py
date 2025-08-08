from setuptools import setup, find_packages
import os
import re

# Read version from version.py
with open(os.path.join(os.path.dirname(__file__), "version.py"), "r") as f:
    version_file = f.read()
    version_match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", version_file)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string in version.py")

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="livekit-voicelab",
    version=version,
    description="VoiceLab TTS plugin for LiveKit Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LiveKit",
    author_email="info@livekit.io",
    url="https://github.com/livekit/voicelab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "aiohttp>=3.8.0",
        "livekit-agents>=0.1.0",
    ],
    keywords="livekit, tts, text-to-speech, voicelab, vogent",
    entry_points={
        "livekit.plugins": [
            "voicelab=voicelab.entry_points:get_plugin",
        ],
    },
)
