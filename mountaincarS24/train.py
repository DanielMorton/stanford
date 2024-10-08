from util import DiscreteGymMDP, ContinuousGymMDP, simulate
from submission import (
    ModelBasedMonteCarlo,
    TabularQLearning,
    FunctionApproxQLearning,
    ConstrainedQLearning,
    fourierFeatureExtractor,
)

import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
import sys, argparse, random, json


def movingAverage(x, window):
    cumSum = np.cumsum(x)
    ma = (cumSum[window:] - cumSum[:-window]) / window
    return ma

def plotRewards(trainRewards, evalRewards, savePath=None, show=True):
    plt.figure(figsize=(10, 5))
    window = 30
    trainMA = movingAverage(trainRewards, window)
    evalMA = movingAverage(evalRewards, window)
    tLen = len(trainRewards)
    eLen = len(evalRewards)
    plt.scatter(range(tLen), trainRewards, alpha=0.5, c='tab:blue', linewidth=0, s=5)
    plt.plot(range(int(window/2), tLen-int(window/2)), trainMA, lw=2, c='b')
    plt.scatter(range(tLen, tLen+eLen), evalRewards, alpha=0.5, c='tab:green', linewidth=0, s=5)
    plt.plot(range(tLen+int(window/2), tLen+eLen-int(window/2)), evalMA, lw=2, c='darkgreen')
    plt.legend(['train rewards', 'train moving average', 'eval rewards', 'eval moving average'])
    plt.xlabel("Episode")
    plt.ylabel("Discounted Reward in Episode")

    if savePath is not None:
        plt.savefig(savePath)
    if show:
        plt.show()



if __name__ == "__main__":
    """
    The main function called when train.py is run
    from the command line:

    > python train.py

    See the usage string for more details.

    > python train.py --help
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["value-iteration", "tabular", "function-approximation", "constrained"],
        help="model-based value iteration (\"value-iteration\"), tabular Q-learning (\"tabular\"), \
            function approximation Q-learning (\"function-approximation\"), or constrained Q-Learning (\"constrained\")",
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        help="Max speed constraint that only applies when doing function approximation",
    )
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    if args.agent == "value-iteration":
        random.seed(2)
        print("************************************************")
        print("Training agent with model-based value iteration to perform mountain car task!")
        print("************************************************")
        mdp = DiscreteGymMDP(
            "MountainCar-v0",
            discount=0.999,
            low=[-1.2, -0.07],
            high=[0.6, 0.07],
            feature_bins=20,
            timeLimit=1000,
        )
        rl = ModelBasedMonteCarlo(
            mdp.actions, mdp.discount, calcValIterEvery=1e5, explorationProb=0.50
        )
        trainRewards = simulate(mdp, rl, train=True, numTrials=3500, verbose=True)
        print("Training complete! Running evaluation, writing weights to mcvi_weights.json and generating reward plot...")
        evalRewards = simulate(mdp, rl, train=False, numTrials=500)

        serialData = {str(key): val for key, val in rl.pi.items()}
        
        with open("mcvi_weights.json", "w") as f:
            json.dump(serialData, f)
        plotRewards(trainRewards, evalRewards, 'mcvi.png')

    # Trained Discrete Agent
    elif args.agent == "tabular":
        print("********************************************************")
        print("Training agent with Tabular Q-Learning to perform mountain car task!")
        print("********************************************************")
        mdp = DiscreteGymMDP(
            "MountainCar-v0",
            discount=0.999,
            low=[-1.2, -0.07],
            high=[0.6, 0.07],
            feature_bins=20,
            timeLimit=1000,
        )
        rl = TabularQLearning(mdp.actions, mdp.discount, explorationProb=0.15)
        trainRewards = simulate(mdp, rl, train=True, numTrials=3500, verbose=True)
        print("Training complete! Running evaluation, writing weights to tabular_weights.npy and generating reward plot...")
        evalRewards = simulate(mdp, rl, train=False, numTrials=500)
        Q = dict(rl.Q)
        np.save("tabular_weights", np.array(Q))

        plotRewards(trainRewards, evalRewards, 'tabular.png')

    # Trained Continuous Agent
    elif args.agent == "function-approximation":
        print("********************************************************")
        print(
            "Training agent with Function Approximation Q-Learning to perform mountain car task!"
        )
        print("********************************************************")
        if args.max_speed is not None:
            gym.register(
                id="CustomMountainCar-v0",
                entry_point="custom_mountain_car:CustomMountainCarEnv",
                max_episode_steps=1000,
                reward_threshold=-110.0,
            )
            mdp = ContinuousGymMDP(
                "CustomMountainCar-v0",
                max_speed=args.max_speed,
                discount=0.999,
                timeLimit=1000,
            )
        else:
            mdp = ContinuousGymMDP("MountainCar-v0", discount=0.999, timeLimit=1000)
        rl = FunctionApproxQLearning(
            36,
            lambda s: fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
            mdp.actions,
            mdp.discount,
            explorationProb=0.2,
        )
        trainRewards = simulate(mdp, rl, train=True, numTrials=3500, verbose=True)
        print("Training complete! Running evaluation, writing weights to fapprox_weights.npy and generating reward plot...")
        evalRewards = simulate(mdp, rl, train=False, numTrials=500)
        np.save("fapprox_weights", rl.W)

        plotRewards(trainRewards, evalRewards, 'fapprox.png')

    elif args.agent == "constrained":
        print("********************************************************")
        print(
            "Training agent with Constrained Q-Learning to perform mountain car task!"
        )
        print("********************************************************")
        gym.register(
            id="CustomMountainCar-v0",
            entry_point="custom_mountain_car:CustomMountainCarEnv",
            max_episode_steps=1000,
            reward_threshold=-110.0,
        )
        mdp = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, timeLimit=1000)
        rl = ConstrainedQLearning(
            36,
            lambda s: fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
            mdp.actions,
            mdp.discount,
            mdp.env.force,
            mdp.env.gravity,
            explorationProb=0.2,
        )
        trainRewards = simulate(mdp, rl, train=True, numTrials=3500, verbose=True)
        print("Training complete! Running evaluation, writing weights to constrainted_weights.npy and generating reward plot...")
        evalRewards = simulate(mdp, rl, train=False, numTrials=500)
        np.save("constrained_weights", rl.W)

        plotRewards(trainRewards, evalRewards, 'constrained.png')
