from setuptools import setup, find_packages

with open("requirements.txt", "r") as req:
    requirements = req.read().splitlines()

setup(
    name='e_maze',
    author="DML group",
    version='0.1',
    description="Benchmark set of environments",
    url="https://github.com/thesmartrobot/ambigym",
    packages=find_packages(),
    python_requires=">3.6",
    install_requires=requirements,
)