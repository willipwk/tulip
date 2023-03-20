from distutils.core import setup

from setuptools import find_packages

dependencies = ["numpy==1.19", "pybullet", "scipy=1.9", "pymeshlab"]

dev_tools = ["ipdb"]


setup(
    name="tulip",
    version="0.0",
    description="Codebase for manipulation projects",
    author="ZENG Yuwei",
    author_email="yuwei.zeng0101@gmail.com",
    license="MIT",
    url="https://www.python.org/sigs/distutils-sig/",
    packages=find_packages("."),
    install_requires=dependencies,
    extras_require={"dev": dev_tools},
)
