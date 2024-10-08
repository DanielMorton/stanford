# Mountaincar

The following sections detail some general notes for working with `mountaincar`, including setup, and
various dependency requirements.

## Prerequisites

Install the following dependencies (in a virtual environment, such as
 [miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers)) for working with OpenAI gym.

```bash
pip3 install -r requirements.txt
```

This command should work out of the box for all platforms (Linux, Mac OS, Windows).

If you are getting "module not found" errors after downloading all the requirements, try using `pip` instead of `pip3` and `python` instead of `python3`. 

### Training agent with your RL Implementation

To train the mountaincar agent with your RL implmentations,

```bash
# MCValueIteration
python3 train.py --agent value-iteration

# Tabular Q-Learning
python3 train.py --agent tabular

# Function Approximation Q-Learning
python3 train.py --agent function-approximation

# Constrained Q-Learning
python3 train.py --agent constrained
```

This will save the resulting policy, Q values, or weights in your assignment directory. 

## Visualizing the Trained Agent

To visualize the agent trained with your RL implementations,

```bash
# Agent trained with MCValueIteration
python3 mountaincar.py --agent value-iteration

# Agent trained with Tabular Q-Learning
python3 mountaincar.py --agent tabular

# Agent trained with Function Approximation Q-Learning
python3 mountaincar.py --agent function-approximation

# Agent trained with Constrained Q-Learning
python3 mountaincar.py --agent constrained
```
