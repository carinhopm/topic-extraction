from setuptools import find_packages, setup
from setuptools.config import read_configuration
from pathlib import Path

conf_dict = read_configuration(Path(__file__).parent / 'setup.cfg')

setup(
    name=conf_dict['metadata']['name'],
    version=conf_dict['metadata']['version'],
    packages=find_packages(),
    install_requires=conf_dict['options']['install_requires'],
    include_package_data=True,
)
