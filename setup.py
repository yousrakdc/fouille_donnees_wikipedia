from setuptools import setup, find_packages

setup(
    name="wikicluster",
    version="1.0.0",
    author="Yousra Kerdouchi",
    author_email="yousra.kerdouchi@example.com",
    description="Clustering thématique non supervisé d'articles scientifiques Wikipedia",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yousrakerdouchi/wikicluster",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.30.0",
        "wikipedia-api>=0.6.0",
        "spacy>=3.7.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "gensim>=4.3.0",
        "minisom>=2.3.0",
    ],
    entry_points={
        "console_scripts": [
            "wikicluster=app:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    license="GPL-3.0",
)