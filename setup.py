from setuptools import setup, find_packages

setup(
    name="evalne",
    version='0.1',
    url="https://bitbucket.org/ghentdatascience/evaluatinggraphembeddings/src/master/",
    license="MIT License",
    author="Alexandru Mara",
    author_email='alexandru.mara@ugent.be',
    description="Open Source Network Embedding Evaluation toolkit",
    packages=find_packages(),
    long_description=open("./README.md").read(),
    zip_safe=False
)
