from setuptools import setup


setup(
    name="surprisemes",
    author="Emiliano Marchese",
    author_email='emilianomarcheserc@gmail.com',
    packages=["surprisemes"],
    package_dir={'': 'src'},
    version="0.1.0",
    description="bla",
    license="MIT",
    install_requires=["numpy>=1.17",
                      "networkx>=2.4",
                      "scipy>=1.4",
                      ],
    extras_require={
        "dev": ["pytest==6.0.1",
                "coverage==5.2.1",
                "pytest-cov==2.10.1",
                "flake8==3.8.3",
                "wheel==0.35.1",
                "matplotlib==3.3.2",
                "check-manifest==0.44",
                "setuptools==47.1.0",
                "twine==3.2.0",
                "tox==3.20.1"],
        },
)
