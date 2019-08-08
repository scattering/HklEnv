from setuptools import setup, find_packages
print("packages:", find_packages())
setup(
    name='HklEnv', 
    packages= find_packages(), 
    version='0.0.1', 
    install_requires=['gym', 'baselines'],
    dependency_links=['git://github.com/scattering/baselines'],
    )

