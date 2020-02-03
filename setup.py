"""Setuptools package definition for PyPI/pip distribution

Dependencies will need to be updated in the "install_requires" of setup()
below. Those dependencies are used to create the CircleCI virtual environment.
These are generally the same dependencies as in environment.yml, but should be
limited to dependencies required by most users. New directories added under
the jiant directory will also need to be added to the "packages" section of
setup().

Distributions are automatically versioned based on git tags. After creating a
new git tag, a release can be created by running:

    # install twine, if necessary
    # pip install --user twine

    # create distribution
    python setup.py sdist bdist_wheel

    # upload to PyPI
    python -m twine upload dist/*

Twine will prompt for login. Login details can be stored for reuse in the file
"~/.pypirc". See https://docs.python.org/3.3/distutils/packageindex.html#pypirc

If you need to test a distribution before tagging, you can use the following
(with example version 0.1.0rc1), but take care to delete the distribution from
dist before the next twine upload to PyPI:

    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0rc1 python setup.py sdist bdist_wheel
    python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jiant",
    author="NYU Machine Learning for Language Group",
    author_email="bowman@nyu.edu",
    description="jiant is a software toolkit for natural language processing research, designed to \
    facilitate work on multitask learning and transfer learning for sentence understanding tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nyu-mll/jiant",
    license="MIT",
    packages=[
        "jiant",
        "jiant.allennlp_mods",
        "jiant.metrics",
        "jiant.modules",
        "jiant.modules.onlstm",
        "jiant.modules.prpn",
        "jiant.huggingface_transformers_interface",
        "jiant.tasks",
        "jiant.utils",
    ],
    install_requires=[
        "torch==1.0.*",
        "numpy==1.14.5",
        "pandas==0.23.0",
        "allennlp==0.8.4",
        "jsondiff",
        "nltk==3.4.5",
        "pyhocon==0.3.35",
        "python-Levenshtein==0.12.0",
        "sacremoses",
        "transformers==2.3.0",
        "ftfy",
        "spacy",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={"": ["jiant/config/**/*.conf"]},
)
