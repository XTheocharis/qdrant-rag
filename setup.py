from setuptools import setup, find_packages

setup(
    name="qdrant_rag",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'pipeline=qdrant_rag:main_cli',
        ],
    },
)
