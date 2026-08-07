# myproject/setup.py

from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="llama-index-vector-stores-endee",
    version="1.1.0",
    packages=find_packages(include=["llama_index_endee", "llama_index_endee.*"]),
    install_requires=[
        "llama-index>=0.12.34",
        "endee>=2.1.0",
        "endee_model",
        "fastembed>=0.3.0",
        "pydantic>=1.9.0,<3",
    ],
    author="Endee Labs",
    author_email="support@endee.io",
    description="Vector Database for Fast ANN Searches",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://endee.io",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
