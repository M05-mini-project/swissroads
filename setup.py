#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages


def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]


setup(
    name="rr_swissroads",
    version="1.0.6",
    description="Basic example of a Reproducible Research Project in Python linked to swissroads mini-project.",
    url="https://github.com/M05-mini-project/swissroads",
    license="BSD",
    author="Christophe Hoël / Paul Arzul",
    author_email="paul.arzul@etu.unidistance.ch",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["rr_swissroads-results = rr_swissroads.results:main"]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
