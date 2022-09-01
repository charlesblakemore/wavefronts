from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
	long_description = fh.read()

setup(name="wavefronts", version=1.0, 
      package_dir={"": "lib"},
      packages=find_packages(), 
      author="Charles Blakemore", 
      author_email="chas.blakemore@gmail.com",
      description="Visualization and Analysis of Measured Optical Wavefronts",
      long_description=long_description,
      url="https://github.com/charlesblakemore/wavefronts")

