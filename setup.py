from setuptools import setup

with open('README.md', "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="surprisememore",
    author="Emiliano Marchese",
    author_email='emilianomarcheserc@gmail.com',
    packages=["surprisememore"],
    package_dir={'': 'src'},
    version="0.1.0",
    description="SurpriseMeMore is a python module providing methods, based on"
                " the surprise framework, to detect mesoscale structures"
                "(e.g. communities, core-periphery, bow-tie) in graphs and "
                "multigraphs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/EmilianoMarchese/SurpriseMeMore",
    download_url="https://github.com/EmilianoMarchese/SurpriseMeMore/archive/refs/tags/v0.1.0.zip",
    keywords=['community detection', 'core-periphery detection',
              'graphs', "multi-graphs", "weighted graphs", "surprise"],
    classifiers=[
                "License :: OSI Approved :: MIT License",
                'Programming Language :: Python :: 3.6',
                'Programming Language :: Python :: 3.7',
                'Programming Language :: Python :: 3.8',
                ],

    install_requires=[
        "numpy>=1.17",
        "networkx>=2.4",
        "scipy>=1.4",
        "numba>=0.47",
        "mpmath>=1.2",
        "tqdm>=4.5"
                      ],
    extras_require={
        "dev": ["pytest>=6.0.1",
                "flake8>=3.8.3",
                "wheel>=0.36.2",
                "check-manifest>=0.44",
                "setuptools>=47.1.0",
                "twine>=3.2.0",
                "tox>=3.20.1"],
        },
)
