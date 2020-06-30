import setuptools

setuptools.setup(
    name="package2vec",
    version="0.0.1",
    author="Jakub Monhart",
    author_email="monhajak@fel.cvut.cz",
    description="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    install_requires = [
        'torch',
        'scikit-learn'
    ]

)