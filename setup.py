# myproject/setup.py

from setuptools import find_packages, setup

setup(
    name="langchain_endee",
    version="1.1.0",
    packages=find_packages(include=["langchain_endee", "langchain_endee.*"]),
    install_requires=[
        # List your dependencies here
        "langchain>=0.3.25",
        "langchain-core>=0.3.59",
        "endee>=1.1.0",
        "endee_model",
        "pydantic>=1.9.0,<3",
    ],
    author="Endee Labs",
    author_email="support@endee.io",
    description=(
        "High Speed Vector Database for Faster and Efficient "
        "ANN Searches with LangChain"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://endee.io",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
