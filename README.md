# HklEnv
An evnvironment for a reinforcement learning approach to faster crystallographic measurements using Proximal Policy Optimization 2 from OpenAI baselines


## Installation

Build a new python environment with the usual numpy, matplotlib, scipy, h5py.  Using an HPC systems with an anaconda module, this would be something like:

    module load anaconda3
    conda create -n hklgym numpy matplotlib scipy h5py

Install the appropriate tensorflow / tensorflow-gpu.  On IBM power architecture, the base tensorflow packages are much too old, but you maybe can use
the IBM powerai anaconda channel:

    conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
    source activate hklgym
    conda install tensorflow-gpu

Install openai gym into your python enviroment as:

    pip install gym

We depend on our own fork of openai baselines, which can be installed from the git repo as:

    git clone https://github.com/scattering/baselines.git
    (cd baselines && pip install -e .)

or directly from pip as:

    pip install "git+https://github.com/scattering/baselines.git#egg=baselines"

Need pycrysfml installed from source:

    git clone https://github.com/scattering/pycrysfml.git
    cd pycrysfml
    # ... may need to modify machine environment in build.sh ...
    ./build.sh
    pip install -e .
    cd ..

Finally, change into this source directory and type:

    git clone https://github.com/scattering/HklEnv.git
    (cd HklEnv && pip install -e .)

which installs hkl as an edittable environment in the gym.

To run HklEnv on a particular problem, define the configuration files in a datapath and set the following os environment variable:

    HKL_DATAPATH=/path/to/problem

Default is "pycrysfml/hklgen/examples/sxtal".

Run a training example on the new hkl environment:

    python3 -m baselines.run --alg=a2c --env=HklEnv:hkl-v0 --num_timesteps=2e16 --log_path=~/logs/hklstest/ --num_env=1

Normally this command would be part of a slurm batch script, run.sh:

    #!/bin/bash
    #SBATCH -c 40
    #SBATCH --gres=gpu:3
    #SBATCH --partition=gpu
    #SBATCH --time=96:00:00
    module load anaconda3
    source activate hklgym
    python3 -m baselines.run --alg=a2c --env=HklEnv:hkl-v0 --num_timesteps=2e16 --log_path=~/logs/hklstest/ --num_env=1

submit using

    sbatch run.sh

## Problems

If you've come to the README because the simple setup.py didn't work, you probably need to uninstall the 
official baselines, and install our forked version. Use the following to uninstall:

    pip uninstall baselines

