from setuptools import find_packages, setup

setup(
    name="heuristic_hafnian",
    packages=find_packages(),
    version="0.0.1",
    install_requires=["numpy", "scipy", "thewalrus", "tqdm", "typer"],
    extras_require={
        "test": ["pytest"],
    },
)
