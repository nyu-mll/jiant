import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jiant",
    author="NYU Machine Learning for Language",
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
        "jiant.openai_transformer_lm",
        "jiant.pytorch_transformers_interface",
        "jiant.tasks",
        "jiant.utils",
    ],
    install_requires=[
        "torch==1.0.*",
        "numpy==1.14.5",
        "pandas==0.23.0",
        "allennlp==0.8.4",
        "jsondiff",
        "nltk==3.2.5",
        "pyhocon==0.3.35",
        "python-Levenshtein==0.12.0",
        "pytorch-transformers==1.0.0",
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
