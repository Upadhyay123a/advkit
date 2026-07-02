from setuptools import setup, find_packages

setup(
    name="advkit",
    version="0.1.0",
    description="Adversarial attack toolkit for image classifiers",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pillow",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "advkit=advkit.cli:main",
        ],
    },
    python_requires=">=3.9",
)