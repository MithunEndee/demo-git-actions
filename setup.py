from setuptools import find_packages, setup

setup(
    name="crewai-endee",
    version="1.1.0",
    packages=find_packages(include=["crewai_endee", "crewai_endee.*"]),
    install_requires=[
        "endee>=2.1.0",
        "endee_model",
        "crewai_tools",
        "crewai",
    ],
    python_requires=">=3.10,<3.14",
    author="Endee Labs",
    author_email="support@endee.io",
    description="Endee vector database integration for CrewAI agent memory",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://endee.io/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
