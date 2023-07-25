from setuptools import find_packages, setup #package and distribute Python projects
from typing import List #annotate function signatures

HYPHEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    # this function reads a file containing requirements for the project and return them as a list.
    requirements=[] # Initialize empty list to store the project requirements 
    with open(file_path) as file_obj:
        #line 9 ensures file is properly closed after reading its contents
        requirements=file_obj.readlines() #reads all the lines from the opened file and assigns them to the requirements variable as a list of strings.
        requirements=[req.replace("\n","") for req in requirements] #removes the newline characters

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements
    

setup(
    name='mlproject',
    version='0.0.1',
    author='bwisa',
    author_email='lynseybwisa@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)