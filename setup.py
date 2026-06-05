"""
Setup configuration for Endee Vector Database Client.

This file defines the package metadata, dependencies, and installation
requirements for the Endee Python client library.
"""

from setuptools import find_packages, setup

# Read the long description from README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


setup(
    # Package Metadata
    name="endee",
    version="0.1.27",
    author="",
    author_email="",
    description=(
        "Endee is the Next-Generation Vector Database for Scalable, High-Performance AI"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    # Package Discovery
    packages=find_packages(),
    # Dependencies
    install_requires=[
        "requests>=2.28.0",  # HTTP library for API requests
        "httpx[http2]>=0.28.1",  # Alternative HTTP library with HTTP/2 support
        "numpy>=2.2.4",  # Array operations and vector normalization
        "msgpack>=1.1.0",  # Efficient binary serialization
        "orjson>=3.11.5",  # Ultra-fast JSON serialization/deserialization
        "pydantic>=1.9.0,<3",  # Data validation and settings management
    ],
    # Python Version Requirements
    python_requires=">=3.6",
    # Package Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # Additional Metadata
    keywords=[
        "vector database",
        "embeddings",
        "machine learning",
        "AI",
        "similarity search",
        "HNSW",
        "nearest neighbors",
    ],
    project_urls={
        "Documentation": "https://docs.endee.io",
    },
)
