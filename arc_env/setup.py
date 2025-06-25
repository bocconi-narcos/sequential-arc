from setuptools import setup, find_packages
import pathlib

# It's good practice to read the version from a single source of truth
# For now, we'll hardcode it here and in pyproject.toml, but ideally,
# it would be read from arc_env/version.py.

VERSION = "0.1.0" # Should match pyproject.toml and eventually arc_env/version.py
PACKAGE_NAME = "arc_env"
DESCRIPTION = "A Gymnasium environment for the Abstraction and Reasoning Corpus (ARC)"
HERE = pathlib.Path(__file__).parent
LONG_DESCRIPTION = (HERE / "README.md").read_text() if (HERE / "README.md").exists() else DESCRIPTION
AUTHOR = "Your Name" # Replace
AUTHOR_EMAIL = "your.email@example.com" # Replace
PROJECT_URLS = {
    "Homepage": "https://github.com/yourusername/arc_env", # Replace
    "Bug Tracker": "https://github.com/yourusername/arc_env/issues", # Replace
}
LICENSE = "MIT" # Choose an appropriate license

# Core dependencies - should align with pyproject.toml
INSTALL_REQUIRES = [
    "gymnasium>=0.26.0",
    "numpy>=1.21.0",
]

# Optional dependencies - should align with pyproject.toml
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0",
        "mypy>=0.900",
        "ruff>=0.1.0",
        "black>=22.0",
        "isort>=5.0",
    ],
    "test": [
        "pytest>=7.0",
        "pytest-cov>=3.0",
    ],
}

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(
        exclude=["tests*", "examples*", "docs*", "scripts*"]
    ),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.8",
    url=PROJECT_URLS.get("Homepage"),
    project_urls=PROJECT_URLS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # If you have entry points, e.g., for command-line scripts
    # entry_points={
    #     "console_scripts": [
    #         "arc-env-cli=arc_env.cli:main",
    #     ],
    # },
)
