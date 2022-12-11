import setuptools
import versioneer
import sys
import os

# ensure the current directory is on sys.path so versioneer can be imported
# when pip uses PEP 517/518 build rules.
# https://github.com/python-versioneer/python-versioneer/issues/193
sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    setuptools.setup(
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
    )
