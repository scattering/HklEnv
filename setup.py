from setuptools import setup, find_packages

packages = find_packages()
setup(
    name='HklEnv', 
    version='0.0.1', 
    packages=packages, 
    install_requires=['gym', 'baselines'],
)
