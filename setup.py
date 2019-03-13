from setuptools import setup, find_packages

setup(
    name="evalne",
    version='0.2.1',
    url="https://github.com/Dru-Mara/EvalNE",
    license="MIT License",
    author="Alexandru Mara",
    author_email='alexandru.mara@ugent.be',
    description="Open Source Network Embedding Evaluation toolkit",
    long_description=open("./README.md").read(),
    long_description_content_type="text/markdown",
    keywords='evaluation embedding link-prediction networks graphs',
    packages=find_packages(),
    python_requires='>=2.6, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*, <4',
    zip_safe=False,
    install_requires=[
        'numpy==1.15.1',
        'scikit-learn>=0.19.0',
        'networkx>=2.2',
        'scipy',
        'matplotlib'
    ],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Intended Audience :: Science/Research"
    ]
)
