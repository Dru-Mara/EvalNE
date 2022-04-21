"""
Setup script for EvalNE. You can install the library globally using:

python setup.py install

Or for a single user with:

python setup.py install --user
"""

from setuptools import setup, find_packages

setup(
    name="evalne",
    version='0.4.0',
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
    python_requires='<3.7',
    zip_safe=False,
    tests_require=["pytest", "pytest-cov"],
    install_requires=[
        'numpy',
        'scikit-learn',
        'networkx',
        'scipy',
        'matplotlib',
        'pandas',
        'pyparsing',
        'tqdm',
        'kiwisolver',
        'joblib'
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ]
)
