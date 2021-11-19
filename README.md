# CvarAdversarialRL
Official code repository for "A Game-Theoretic Perspective on Risk-Sensitive Reinforcement Learning".

## Initial setup

Create a virtual environment using

```python3 -m venv ${YOUR_VENVS_DIR}/cvarRL```

and activate it

```source ${YOUR_VENVS_DIR}/cvarRL/bin/activate```

Install the necessary requirements

```pip3 install -r requirements.txt```

Add the current folder to your PYTHONPATH

```export PYTHONPATH="${PYTHONPATH}:${YOUR_PARENT_DIR}/CvarAdversarialRL"```

## Running the experiments and collecting figures

Scripts are produced to allow easy reproductibility of our results.
They can be found in the scripts folder.

To run experiments:

```./scripts/run_experiments.sh```

To generate figures:

```./scripts/generate_figures.sh```
