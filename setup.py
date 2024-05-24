from setuptools import setup, find_packages

setup(
    name="mdod",
    version="0.1.9",
    packages=find_packages(),
    package_data={"": ["*"]},  
    install_requires=[
        '',
    ],
    author="Z Shen",
    author_email="626456708@qq.com",
    description="MDOD, Multi-Dimensional data Outlier Detection",
    long_description="Python library for Multi-Dimensional data Outlier/Anomaly Detection algorithm. For details please read README.md, or visit https://github.com/mddod/mdod",
    license="BSD 3-Clause License",
    url="https://github.com/mddod/mdod",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)
