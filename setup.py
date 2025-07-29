from setuptools import setup, find_packages

setup(
    name="my_imputers",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn"
    ],
    author="Your Name",
    description="A custom sklearn-compatible imputer using model-based predictions",
    python_requires=">=3.8",
)
