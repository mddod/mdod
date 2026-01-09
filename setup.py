from setuptools import setup, find_packages

#  README.md 
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mdod",
    version="3.0.6.1",  # version
    packages=find_packages(),

    # requires
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],

    # author
    author="Z Shen",
    author_email="626456708@qq.com",

    # description
    description="Multi-Dimensional Outlier Detection (MDOD) using vector cosine similarity with added virtual dimension",

    # long_description（README.md）
    long_description=long_description,
    long_description_content_type="text/markdown",

    # url
    url="https://github.com/mddod/mdod",
    project_urls={
        "Bug Tracker": "https://github.com/mddod/mdod/issues",
        "Documentation": "https://mddod.github.io/",
        "Source Code": "https://github.com/mddod/mdod",
        "Paper": "https://doi.org/10.48550/arXiv.2601.00883",
    },

    # keywords
    keywords=[
        "outlier detection",
        "anomaly detection",
        "multi-dimensional data",
        "cosine similarity",
        "machine learning",
        "data mining",
        "unsupervised learning",
        "scikit-learn",
        "python",
        "AI",
    ],

    # Python
    python_requires=">=3.7",

    # license
    license="BSD 3-Clause License",

    # classifiers
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)