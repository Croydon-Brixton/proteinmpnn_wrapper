# Setup code adapted from https://github.com/biotite-dev/biotite/blob/master/setup.py
import re
from os.path import join, abspath, dirname, normpath
import fnmatch
import os
from setuptools import setup, find_packages, Extension
import subprocess

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

# Path to ProteinMPNN
proteinmpnn_path = os.path.join("src", "proteinmpnn", "ProteinMPNN")
proteinmpnn_init_path = os.path.join(proteinmpnn_path, '__init__.py')
# Try calling git submodule update --init --recursive if ProteinMPNN submodule is not found
if not os.path.exists(proteinmpnn_init_path):
    try:
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)
    except subprocess.CalledProcessError as e:
        raise FileNotFoundError(f"ProteinMPNN submodule not found at {proteinmpnn_path}. Call `git submodule update --init --recursive` to clone the submodule first, then try again.") from e
assert os.path.exists(proteinmpnn_path), f"ProteinMPNN directory not found at {proteinmpnn_path} and could not be autocreated. Call `git submodule update --init --recursive` to clone the submodule first, then try again."
# ... create __init__.py in ProteinMPNN submodule clone so it can be found
if not os.path.exists(proteinmpnn_init_path):
    with open(proteinmpnn_init_path, 'w') as f:
        pass  # just to create an empty __init__.py file

setup(
    name="proteinmpnn",
    version = version,
    zip_safe = False,
    packages = find_packages("src"),
    package_dir = {"" : "src"},
    include_package_data=True
)


# Return to original directory
os.chdir(original_wd)