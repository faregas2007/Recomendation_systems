from pathlib import Path
from setuptools import setup

with open('requirements.txt') as f:
    required_packages = [ln.strip() for ln in f.readlines()]

def readme():
    with open('README.md') as f:
        return f.read()

dev_packages=[
    'jupyterlab==2.2.8',
    'flake8==3.8.3',
    'pre-commit==2.11.1',
]

docs_packages=[
    'mkdocs==1.1.2',
    'mkdocstrings==0.14.0',
]

setup(
    name='recsys',
    version='0.1',
    description='recommendation system with MLOps',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    author="Tien Dat Nguyen",
    author_email="faregas2007@gmail.com",
    license='MIT',
    url="https://github.com/faregas2007/Recomendation_systems.git",

    packages=['recsys'],
    python_requires=">=3.6",
    install_requires= [required_packages],
    extras_require={
        'dev': dev_packages + docs_packages,
        'docs': docs_packages,
    },

    entry_points = {
        'console_scripts':[
            'recsys = app.cli:app',
        ],
    },


)