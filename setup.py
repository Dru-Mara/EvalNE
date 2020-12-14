"""
Setup script for EvalNE. You can install the library globally using:

python setup.py install

Or for a single user with:

python setup.py install --user
"""

from setuptools import setup, find_packages
import sys

if sys.version_info[0] == 2:
    alternative = 'kiwisolver==1.1.0'
else:
    alternative = 'kiwisolver==1.3.1'

setup(
    name="evalne",
    version='0.3.3',
    url="https://github.com/Dru-Mara/EvalNE",
    license="MIT License",
    author="Alexandru Mara",
    author_email='alexandru.mara@ugent.be',
    description="Open Source Network Embedding Evaluation toolkit",
    long_description=open("./README.md").read(),
    long_description_content_type="text/markdown",
    keywords='evaluation embedding link-prediction sign-prediction node-classification network-reconstruction '
             'networks graphs visualization',
    packages=find_packages(),
    python_requires='>=2.6, <=3.6',
    zip_safe=False,
    tests_require=["pytest", "pytest-cov"],
    install_requires=[
        'numpy==1.15.4',
        'scikit-learn==0.19.0',
        'networkx==2.2',
        'scipy==0.19.1',
        'matplotlib==2.2.4',
        'pandas==0.24.2',
        'pyparsing==2.4.7',
        'tqdm',
        alternative
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ]
)
