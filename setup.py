import os

from setuptools import setup, find_packages

package_path = os.path.join(".")

setup(
    name="py_amy",
    version="0.1",
    author="Thorsten Greiner",
    author_email="thorsten.greiner@posteo.de",
    url="https://github.com/thgreiner/py-amy",
    packages=["py_amy.engine"],
)
