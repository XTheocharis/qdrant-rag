from setuptools import setup, find_packages

setup(
    name="qdrant-rag",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'pipeline=qdrant-rag:main_cli',
        ],
    },
)
