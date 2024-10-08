from util import DiscreteGymMDP, ContinuousGymMDP, RandomAgent, simulate
from submission import (
    ModelBasedMonteCarlo,
    TabularQLearning,
    FunctionApproxQLearning,
    ConstrainedQLearning,
    fourierFeatureExtractor,
)
from collections import defaultdict

import numpy as np
import gymnasium as gym
import sys, argparse, json

if __name__ == "__main__":
    """
    The main function called when mountaincar.py is run
    from the command line:

    > python mountaincar.py

    See the usage string for more details.

    > python mountaincar.py --help
    """
    # play.play(gym.make("MountainCar-v0", render_mode="human"), zoom=3)
    # TODO: Implement interactive mode for human play
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["naive", "value-iteration", "tabular", "function-approximation", "constrained"],
        help="naive (\"naive\"), model-based value iteration (\"value-iteration\"), tabular Q-learning (\"tabular\"), \
            function approximation Q-learning (\"function-approximation\"), or constrained Q-Learning (\"constrained\")",
    )
    args = parser.parse_args()

    # Naive Agent
    if args.agent == "naive":
        print("************************************************")
        print("Naive agent performing mountain car task!")
        print("************************************************")
        mdp = DiscreteGymMDP("MountainCar-v0", discount=0.999, timeLimit=1000)
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = RandomAgent(mdp.actions)
        simulate(mdp, rl, train=False, numTrials=1, verbose=False, demo=True)
        mdp.env.close()

    # Agent Trained w/ Model-Based Value Iteration
    elif args.agent == "value-iteration":
        print("********************************************************")
        print("Agent trained with model-based value iteration performing mountain car task!")
        print("********************************************************")
        mdp = DiscreteGymMDP(
            "MountainCar-v0",
            discount=0.999,
            low=[-1.2, -0.07],
            high=[0.6, 0.07],
            feature_bins=20,
            timeLimit=1000,
        )
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = ModelBasedMonteCarlo(
            mdp.actions, mdp.discount, calcValIterEvery=1e5, explorationProb=0.15
        )
        with open("mcvi_weights.json", "r") as f:
            data = json.load(f)
        rl.pi = {eval(key): val for key, val in data.items()}
        simulate(mdp, rl, train=False, numTrials=1, verbose=False, demo=True)
        mdp.env.close()

    # Agent Trained w/ Tabular Q-Learning
    elif args.agent == "tabular":
        print("********************************************************")
        print("Agent trained with Tabular Q-Learning performing mountain car task!")
        print("********************************************************")
        mdp = DiscreteGymMDP(
            "MountainCar-v0",
            discount=0.999,
            low=[-1.2, -0.07],
            high=[0.6, 0.07],
            feature_bins=20,
            timeLimit=1000,
        )
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = TabularQLearning(mdp.actions, mdp.discount, explorationProb=0.15)
        Qval = np.load("tabular_weights.npy", allow_pickle=True)
        Qnew = defaultdict(lambda: 0)
        Qnew.update(Qval.item())
        rl.Q = Qnew
        simulate(mdp, rl, train=False, numTrials=1, verbose=False, demo=True)
        mdp.env.close()

    # Agent Trained w/ Function Approx Q-Learning
    elif args.agent == "function-approximation":
        print("********************************************************")
        print(
            "Agent trained with Function Approximation Q-Learning performing mountain car task!"
        )
        print("********************************************************")
        mdp = ContinuousGymMDP("MountainCar-v0", discount=0.999, timeLimit=1000)
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = FunctionApproxQLearning(
            36,
            lambda s: fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
            mdp.actions,
            mdp.discount,
            explorationProb=0.2,
        )
        rl.W = np.load("fapprox_weights.npy", allow_pickle=True)
        simulate(mdp, rl, train=False, numTrials=1, verbose=False, demo=True)
        mdp.env.close()

    # Agent Trained w/ Constrained Q-Learning
    elif args.agent == "constrained":
        print("********************************************************")
        print("Agent trained with Constrained Q-Learning performing mountain car task!")
        print("********************************************************")
        mdp = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, timeLimit=1000)
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = ConstrainedQLearning(
            36,
            lambda s: fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
            mdp.actions,
            mdp.discount,
            mdp.env.force,
            mdp.env.gravity,
            explorationProb=0.2,
        )
        rl.W = np.load("constrained_weights.npy", allow_pickle=True)
        simulate(mdp, rl, train=False, numTrials=1, verbose=False, demo=True)
        mdp.env.close()
