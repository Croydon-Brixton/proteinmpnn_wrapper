# Setup code adapted from https://github.com/biotite-dev/biotite/blob/master/setup.py
import re
from os.path import join, abspath, dirname, normpath
import fnmatch
import os
from setuptools import setup, find_packages, Extension

original_wd = os.getcwd()
# Change directory to setup directory to ensure correct file identification
os.chdir(dirname(abspath(__file__)))

# Parse the top level package for the version
# Do not use an import to prevent side effects
# e.g. required runtime dependencies
version = None
with open(join("src", "__init__.py")) as init_file:
    for line in init_file.read().splitlines():
        if line.lstrip().startswith("__version__"):
            version_match = re.search('".*"', line)
            if version_match:
                # Remove quotes
                version = version_match.group(0)[1 : -1]
            else:
                raise ValueError("No version is specified in '__init__.py'")
if version is None:
    raise ValueError("Unable to identify 'version' in __init__.py")

setup(
    name="proteinmpnn",
    version = version,
    zip_safe = False,
    packages = find_packages("src"),
    package_dir = {"" : "src"},
    
    # Including additional data
    package_data = {
        # TODO: Include ProteinMPNN weights
    },
)


# Return to original directory
os.chdir(original_wd)