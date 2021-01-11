from setuptools import find_packages, setup
setup(
    name="surprisemes",
    packages=find_packages(include=[‘mypythonlib’]),
    version=’0.1.0',
    description=’My first Python library’,
    author=’Me’,
    license=’MIT’,
    install_requires=[],
    setup_requires=["pytest-runner",
                    "numpy",
                    "numba",
                    ],
    tests_require=["pytest==4.4.1",
                   "numpy",
                   "numba",
                    ],
    test_suite=’tests’,
)