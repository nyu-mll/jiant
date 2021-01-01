"""
Simple check list from huggingface/transformers repo: https://github.com/huggingface/transformers/blob/master/setup.py
To create the package for pypi.
1. Change the version in setup.py and docs (if applicable).
2. Unpin specific versions from setup.py.
2. Commit these changes with the message: "Release: VERSION"
3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master
4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).
   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).
   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.
5. Check that everything looks correct by uploading the package to the pypi test server:
   python3 -m twine upload --repository testpypi dist/*
   Alternatively:
   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi jiant
6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi
7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.
8. Add the release version to docs deployment (if applicable)
9. Update README.md to redirect to correct documentation.
"""

import shutil
from pathlib import Path

from setuptools import find_packages, setup

extras = {}
extras["testing"] = ["pytest", "pytest-cov", "pre-commit"]
extras["docs"] = ["sphinx"]
extras["quality"] = ["black == 19.10b0", "flake8-docstrings == 1.5.0", "flake8 >= 3.7.9", "mypy == 0.770"]
extras["dev"] = extras["testing"] + extras["quality"]

setup(
    name="jiant",
    version="2.1.2",
    author="NYU Machine Learning for Language Group",
    author_email="bowman@nyu.edu",
    description="State-of-the-art Natural Language Processing toolkit for multi-task and transfer learning built on PyTorch.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning transformer pytorch tensorflow BERT GPT GPT-2 google nyu datasets",
    license="MIT",
    url="https://github.com/nyu-mll/jiant",
    packages=find_packages(exclude=["tests", "tests/*"]),
    install_requires=[
        "attrs == 19.3.0",
        "bs4 == 0.0.1",
        "jsonnet == 0.15.0",
        "lxml == 4.5.1",
        "datasets == 1.1.2",
        "nltk >= 3.5",
        "numexpr == 2.7.1",
        "numpy == 1.18.4",
        "pandas == 1.0.3",
        "python-Levenshtein == 0.12.0",
        "sacremoses == 0.0.43",
        "seqeval == 0.0.12",
        "scikit-learn == 0.22.2.post1",
        "scipy == 1.4.1",
        "sentencepiece == 0.1.86",
        "tokenizers == 0.8.1.rc2",
        "torch >= 1.5.0",
        "tqdm == 4.46.0",
        "transformers == 3.1.0",
        "torchvision == 0.6.0",
    ],
    extras_require=extras,
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
