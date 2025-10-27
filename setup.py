from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    '''
    This function returns a list of requirements
    '''
    requirements = []
    try:
        with open('requirements.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if line and line != "-e .":
                    requirements.append(line)
    except FileNotFoundError:
        print("="*35)
        print("requirements.txt not found")
        print("="*35)
    except Exception as e:
        print("="*35)
        print(f"An exception occurred: {e}")
        print("="*35)
    return requirements


setup(
    name='network_security',
    version='0.0.1',
    author='Pradyumn Bisht',
    author_email='pradybisht@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)
