import setuptools

# read the requirements.txt file
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# setup the package
setuptools.setup(
    name="iris_toy_example",
    version="0.0.1",
    author="Nikita Bortych",
    packages=requirements)