# HklEnv
An evnvironment for a reinforcement learning approach to faster crystallographic measurements using Proximal Policy Optimization 2 from OpenAI baselines


## Installation

Install the usual numpy, matplotlib, scipy, h5py environment.

Install the appropriate tensorflow / tensorflow-gpu.

Install openai gym as:

    pip install gym

We depend on our own fork of openai baselines, which can be installed from the git repo as:

    git clone https://github.com/scattering/baselines.git
    pip install -e .

or directly from pip as:

    pip install "git+https://github.com/scattering/baselines.git#egg=baselines"

Need pycrysfml installed from source:

    git clone https://github.com/scattering/pycrysfml.git
    git checkout python3
    cd pycrysfml
    # ... may need to modify machine environment in build.sh ...
    ./build.sh
    pip install -e .

Finally, change into this source directory and type:

    git clone https://github.com/scattering/HklEnv.git
    pip install -e .

which installs hkl as an edittable environment in the gym.

To run HklEnv on a particular problem, define the configuration files in a datapath and set the following os environment variable:

    HKL_DATAPATH=/path/to/problem

Default is "pycrysfml/hklgen/examples/sxtal".

## Problems

If you've come to the README because the simple setup.py didn't work, you probably need to uninstall the 
official baselines, and install our forked version. Use the following to uninstall:

    pip uninstall baselines

