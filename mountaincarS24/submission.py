import math, random
from collections import defaultdict
from typing import List, Callable, Tuple, Dict, Any, Optional, Iterable, Set
import gymnasium as gym
import numpy as np

import util
from util import ContinuousGymMDP, StateT, ActionT
from custom_mountain_car import CustomMountainCarEnv

############################################################
# Problem 3a
# Implementing Value Iteration on Number Line (from Problem 1)
def valueIteration(succAndRewardProb: Dict[Tuple[StateT, ActionT], List[Tuple[StateT, float, float]]], discount: float, epsilon: float = 0.001):
    '''
    Given transition probabilities and rewards, computes and returns V and
    the optimal policy pi for each state.
    - succAndRewardProb: Dictionary mapping tuples of (state, action) to a list of (nextState, prob, reward) Tuples.
    - Returns: Dictionary mapping each state to an action.
    '''
    # Define a mapping from states to Set[Actions] so we can determine all the actions that can be taken from s.
    # You may find this useful in your approach.
    stateActions = defaultdict(set)
    for state, action in succAndRewardProb.keys():
        stateActions[state].add(action)

    def computeQ(V: Dict[StateT, float], state: StateT, action: ActionT) -> float:
        # Return Q(state, action) based on V(state)
        return sum(prob * (reward + discount * V[nextState])
                   for nextState, prob, reward in succAndRewardProb[(state, action)])

    def computePolicy(V: Dict[StateT, float]) -> Dict[StateT, ActionT]:
        policy = {}
        for state in stateActions:
            q_values = [(action, computeQ(V, state, action)) for action in stateActions[state]]
            policy[state] = max(q_values, key=lambda x: (x[1], x[0]))[0]
        return policy

    print('Running valueIteration...')
    V = defaultdict(float) # This will return 0 for states not seen (handles terminal states)
    numIters = 0
    while True:
        newV = defaultdict(float)
        for state in stateActions:
            newV[state] = max(computeQ(V, state, action) for action in stateActions[state])
        if all(abs(newV[s] - V[s]) < epsilon for s in stateActions):
            break
        V = newV
        numIters += 1
    V_opt = V
    print(("valueIteration: %d iterations" % numIters))
    return computePolicy(V_opt)

############################################################
# Problem 3b
# Model-Based Monte Carlo

# Runs value iteration algorithm on the number line MDP
# and prints out optimal policy for each state.
def run_VI_over_numberLine(mdp: util.NumberLineMDP):
    succAndRewardProb = {
        (-mdp.n + 1, 1): [(-mdp.n + 2, 0.2, mdp.penalty), (-mdp.n, 0.8, mdp.leftReward)],
        (-mdp.n + 1, 2): [(-mdp.n + 2, 0.3, mdp.penalty), (-mdp.n, 0.7, mdp.leftReward)],
        (mdp.n - 1, 1): [(mdp.n - 2, 0.8, mdp.penalty), (mdp.n, 0.2, mdp.rightReward)],
        (mdp.n - 1, 2): [(mdp.n - 2, 0.7, mdp.penalty), (mdp.n, 0.3, mdp.rightReward)]
    }

    for s in range(-mdp.n + 2, mdp.n - 1):
        succAndRewardProb[(s, 1)] = [(s+1, 0.2, mdp.penalty), (s - 1, 0.8, mdp.penalty)]
        succAndRewardProb[(s, 2)] = [(s+1, 0.3, mdp.penalty), (s - 1, 0.7, mdp.penalty)]

    pi = valueIteration(succAndRewardProb, mdp.discount)
    return pi


class ModelBasedMonteCarlo(util.RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float, calcValIterEvery: int = 10000,
                 explorationProb: float = 0.2,) -> None:
        self.actions = actions
        self.discount = discount
        self.calcValIterEvery = calcValIterEvery
        self.explorationProb = explorationProb
        self.numIters = 0

        # (state, action) -> {nextState -> ct} for all nextState
        self.tCounts = defaultdict(lambda: defaultdict(int))
        # (state, action) -> {nextState -> totalReward} for all nextState
        self.rTotal = defaultdict(lambda: defaultdict(float))

        self.pi = {} # Optimal policy for each state. state -> action

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # Should return random action if the given state is not in self.pi.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always follow the policy if available.
    # HINT: Use random.random() (not np.random()) to sample from the uniform distribution [0, 1]
    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4:  # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e6:  # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        if not explore or random.random() > explorationProb:
            if state in self.pi:
                return self.pi[state]
        return random.choice(self.actions)

    # We will call this function with (s, a, r, s'), which is used to update tCounts and rTotal.
    # For every self.calcValIterEvery steps, runs value iteration after estimating succAndRewardProb.
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool):
        self.tCounts[(state, action)][nextState] += 1
        self.rTotal[(state, action)][nextState] += reward

        if self.numIters % self.calcValIterEvery == 0:
            succAndRewardProb = defaultdict(list)
            for (s, a), nextStates in self.tCounts.items():
                total_count = sum(nextStates.values())
                for ns, count in nextStates.items():
                    prob = count / total_count
                    avg_reward = self.rTotal[(s, a)][ns] / count
                    succAndRewardProb[(s, a)].append((ns, prob, avg_reward))

            self.pi = valueIteration(succAndRewardProb, self.discount)

############################################################
# Problem 4a
# Performs Tabular Q-learning. Read util.RLAlgorithm for more information.
class TabularQLearning(util.RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float,
                 explorationProb: float = 0.2, initialQ: float = 0):
        '''
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - explorationProb: the epsilon value indicating how frequently the policy returns a random action
        - intialQ: the value for intializing Q values.
        '''
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.Q = defaultdict(lambda: initialQ)
        self.numIters = 0

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    # HINT 1: You can access Q-value with self.Q[state, action]
    # HINT 2: Use random.random() to sample from the uniform distribution [0, 1]
    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4:  # explore
            explorationProb = 1.0
        elif self.numIters > 1e5:  # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        if not explore or random.random() > explorationProb:
            return max(self.actions, key=lambda a: self.Q[state, a])
        else:
            return random.choice(self.actions)

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.1

    # We will call this function with (s, a, r, s'), which you should use to update |Q|.
    # Note that if s is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update the Q values using self.getStepSize() 
    # HINT 1: The target V for the current state is a combination of the immediate reward
    # and the discounted future value.
    # HINT 2: V for terminal states is 0
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: float, nextState: StateT,
                            terminal: bool) -> None:
        if terminal:
            target = reward
        else:
            target = reward + self.discount * max(self.Q[nextState, a] for a in self.actions)

        stepSize = self.getStepSize()
        oldQ = self.Q[state, action]
        self.Q[state, action] = oldQ + stepSize * (target - oldQ)

############################################################
# Problem 4b: Fourier feature extractor

def fourierFeatureExtractor(
        state: StateT,
        maxCoeff: int = 5,
        scale: Optional[Iterable] = None
) -> np.ndarray:
    if scale is None:
        scale = np.ones_like(state)

    # Create coefficients array
    coeffs = np.arange(maxCoeff + 1)

    # Scale the state
    scaled_state = np.array(state) * scale

    # Create meshgrid of coefficients
    coeff_grid = np.array(np.meshgrid(*[coeffs for _ in state])).T.reshape(-1, len(state))

    # Compute dot product between coefficients and scaled state
    dot_product = np.dot(coeff_grid, scaled_state)

    # Compute cosine features
    features = np.cos(np.pi * dot_product)

    return features

############################################################
# Problem 4c: Q-learning with Function Approximation
# Performs Function Approximation Q-learning. Read util.RLAlgorithm for more information.
class FunctionApproxQLearning(util.RLAlgorithm):
    def __init__(self, featureDim: int, featureExtractor: Callable, actions: List[int],
                 discount: float, explorationProb=0.2):
        '''
        - featureDim: the dimensionality of the output of the feature extractor
        - featureExtractor: a function that takes a state and returns a numpy array representing the feature.
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - explorationProb: the epsilon value indicating how frequently the policy returns a random action
        '''
        self.featureDim = featureDim
        self.featureExtractor = featureExtractor
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.W = np.random.standard_normal(size=(featureDim, len(actions)))
        self.numIters = 0

    def getQ(self, state: np.ndarray, action: int) -> float:
        features = self.featureExtractor(state)
        return np.dot(features, self.W[:, action])

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    # HINT: This function should be the same as your implementation for 4a.
    def getAction(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4:  # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e5:  # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        if not explore or random.random() > explorationProb:
            return max(self.actions, key=lambda a: self.getQ(state, a))
        else:
            return random.choice(self.actions)

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.005 * (0.99)**(self.numIters / 500)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update W using self.getStepSize()
    # HINT 1: this part will look similar to 4a, but you are updating self.W
    # HINT 2: review the function approximation module for the update rule
    def incorporateFeedback(self, state: np.ndarray, action: int, reward: float, nextState: np.ndarray,
                            terminal: bool) -> None:
        features = self.featureExtractor(state)

        if terminal:
            target = reward
        else:
            target = reward + self.discount * max(self.getQ(nextState, a) for a in self.actions)

        prediction = self.getQ(state, action)
        stepSize = self.getStepSize()

        # Update weights
        self.W[:, action] += stepSize * (target - prediction) * features

############################################################
# Problem 5c: Constrained Q-learning

class ConstrainedQLearning(FunctionApproxQLearning):
    def __init__(self, featureDim: int, featureExtractor: Callable, actions: List[int],
                 discount: float, force: float, gravity: float,
                 max_speed: Optional[float] = None,
                 explorationProb=0.2):
        super().__init__(featureDim, featureExtractor, actions,
                         discount, explorationProb)
        self.force = force
        self.gravity = gravity
        self.max_speed = max_speed

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action that is valid.
    def getAction(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4:  # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e5:  # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        position, velocity = state

        def is_action_valid(action):
            force = (action - 1) * self.force
            acceleration = force / 1.0 - self.gravity * np.sin(position)
            new_velocity = velocity + 0.02 * acceleration
            if self.max_speed is not None:
                new_velocity = np.clip(new_velocity, -self.max_speed, self.max_speed)
            new_position = position + 0.02 * new_velocity
            return -np.pi / 2 <= new_position <= np.pi / 2

        valid_actions = [a for a in self.actions if is_action_valid(a)]

        if not explore or random.random() > explorationProb:
            return max(valid_actions, key=lambda a: self.getQ(state, a))
        else:
            return random.choice(valid_actions)

############################################################
# This is helper code for comparing the predicted optimal
# actions for 2 MDPs with varying max speed constraints
gym.register(
    id="CustomMountainCar-v0",
    entry_point="custom_mountain_car:CustomMountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)

mdp1 = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, timeLimit=1000)
mdp2 = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, timeLimit=1000)

# This is a helper function for 5c. This function runs
# ConstrainedQLearning, then simulates various trajectories through the MDP
# and compares the frequency of various optimal actions.
def compare_MDP_Strategies(mdp1: ContinuousGymMDP, mdp2: ContinuousGymMDP):
    rl1 = ConstrainedQLearning(
        36,
        lambda s: fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
        mdp1.actions,
        mdp1.discount,
        mdp1.env.force,
        mdp1.env.gravity,
        10000,
        explorationProb=0.2,
    )
    rl2 = ConstrainedQLearning(
        36,
        lambda s: fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
        mdp2.actions,
        mdp2.discount,
        mdp2.env.force,
        mdp2.env.gravity,
        0.065,
        explorationProb=0.2,
    )
    sampleKRLTrajectories(mdp1, rl1)
    sampleKRLTrajectories(mdp2, rl2)

def sampleKRLTrajectories(mdp: ContinuousGymMDP, rl: ConstrainedQLearning):
    accelerate_left, no_accelerate, accelerate_right = 0, 0, 0
    for n in range(100):
        traj = util.sample_RL_trajectory(mdp, rl)
        accelerate_left = traj.count(0)
        no_accelerate = traj.count(1)
        accelerate_right = traj.count(2)

    print(f"\nRL with MDP -> start state:{mdp.startState()}, max_speed:{rl.max_speed}")
    print(f"  *  total accelerate left actions: {accelerate_left}, total no acceleration actions: {no_accelerate}, total accelerate right actions: {accelerate_right}")
