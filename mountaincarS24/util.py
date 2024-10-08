import collections, random, time
from typing import List, Tuple, Dict, Any, Union, Optional, Iterable
import gymnasium as gym
import numpy as np

StateT = Union[int, float, Tuple[Union[float, int]]]
ActionT = Any

def create_bins(low: List[float], high: List[float], num_bins: Union[int, List[int]]) -> List[np.ndarray]:
    """
    Takes in a gym.spaces.Box and returns a set of bins per feature according to num_bins
    """
    assert len(low) == len(high)
    if isinstance(num_bins, int):
        num_bins = [num_bins for _ in range(len(low))]
    assert len(num_bins) == len(low)
    bins = []
    for low, high, n in zip(low, high, num_bins):
        bins.append(np.linspace(low, high, n))
    return bins

def discretize(x, bins) -> Tuple[int]:
    """
    Discretize an array x according to bins
    x: np.ndarray, shape (features,)
    bins: np.ndarray, shape (features, bins)
    """
    return tuple(int(np.digitize(feature, bin)) for feature, bin in zip(x, bins))

# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")
    
    # Property holding the set of possible actions at each state.
    @property
    def actions(self) -> List[ActionT]: raise NotImplementedError("Override me")

    # Property holding the discount factor
    @property
    def discount(self): raise NotImplementedError("Override me")

    # property holding the maximum number of steps for running the simulation.
    @property
    def timeLimit(self) -> int: raise NotImplementedError("Override me")

    # Transitions the MDP
    def transition(self, action): raise NotImplementedError("Override me")
        
class NumberLineMDP(MDP):
    def __init__(self, leftReward: float = 10, rightReward: float = 50, penalty: float = -5, n: int = 2): 
        self.leftReward = leftReward
        self.rightReward = rightReward
        self.penalty = penalty
        self.n = n
        self.terminalStates = {-n, n}
    
    def startState(self): 
        self.state = 0
        return self.state
    
    @property
    def actions(self): 
        return [1, 2]
    
    def transition(self, action) -> Tuple[StateT, float, bool]:
        assert self.state not in self.terminalStates, "Attempting to call transition on a terminated MDP."
        if action == 1:
            forward_prob = 0.2
        elif action == 2:
            forward_prob = 0.3
        else:
            raise ValueError("Invalid Action Provided.")
        
        if random.random() < forward_prob:
            # Move the agent forward
            self.state += 1
        else:
            # Move the agent backward
            self.state -= 1

        if self.state == self.n:
            reward = self.rightReward
        elif self.state == -self.n:
            reward = self.leftReward
        else:
            reward = self.penalty
        
        # Check for termination
        terminal = self.state in self.terminalStates
        
        return (self.state, reward, terminal)
    
    @property
    def discount(self): 
        return 1.0

class GymMDP(MDP):
    def __init__(self, env, max_speed: Optional[float] = None, discount: float = 0.99, timeLimit: Optional[int] = None):
        self.max_speed = max_speed
        if self.max_speed is not None:
            self.env = gym.make(env, max_speed=self.max_speed)
        else:
            self.env = gym.make(env)
        assert isinstance(self.env.action_space, gym.spaces.Discrete), "Must use environments with discrete actions"
        assert isinstance(self.env, gym.wrappers.TimeLimit)
        if timeLimit is not None:
            self.env._max_episode_steps = timeLimit
        self._time_limit = self.env._max_episode_steps
        self._discount = discount
        self._actions = list(range(self.env.action_space.n))
        self._reset_seed_gen = np.random.default_rng(0)

    # Return the number of steps before the MDP should be reset.
    @property
    def timeLimit(self) -> int:
        return self._time_limit

    # Return set of actions possible at every state.
    @property
    def actions(self) -> List[ActionT]:
        return self._actions

    # Return the MDP's discount factor
    @property
    def discount(self):
        return self._discount

    # Returns the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Returns a tuple of (nextState, reward, terminated)
    def transition(self, action): raise NotImplementedError("Override me")

class ContinuousGymMDP(GymMDP):

    def startState(self):
        state, _ = self.env.reset(seed=int(self._reset_seed_gen.integers(0, 1e6)))
        return state

    def transition(self, action):
        nextState, reward, terminal, _, _ = self.env.step(action)
        return (nextState, reward, terminal)

class DiscreteGymMDP(GymMDP):

    def __init__(self, env, feature_bins: Union[int, List[int]] = 10, low: Optional[List[float]] = None, high: Optional[List[float]] = None, **kwargs):
        super().__init__(env, **kwargs)
        assert isinstance(self.env.observation_space, gym.spaces.Box) and len(self.env.observation_space.shape) == 1
        low = self.env.observation_space.low if low is None else low
        high = self.env.observation_space.high if high is None else high
        # Convert the environment to a discretized version
        self.bins = create_bins(low, high, feature_bins)

    def startState(self):
        state, _ = self.env.reset(seed=int(self._reset_seed_gen.integers(0, 1e6)))
        return discretize(state, self.bins)

    def transition(self, action):
        nextState, reward, terminal, _, _ = self.env.step(action)
        nextState = discretize(nextState, self.bins)
        return (nextState, reward, terminal)

############################################################

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state: StateT) -> ActionT: raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |nextState|.
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool):
        raise NotImplementedError("Override me")

# An RL algorithm that acts according to a fixed policy |pi| and doesn't
# actually do any learning.
class FixedRLAlgorithm(RLAlgorithm):
    def __init__(self, pi: Dict[StateT, ActionT], actions: List[ActionT], explorationProb: float = 0.2): 
        self.pi = pi
        self.actions = actions
        self.explorationProb = explorationProb

    # Just return the action given by the policy.
    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore and random.random() < self.explorationProb:
            return random.choice(self.actions)
        else:
            return self.pi[state]

    # Don't do anything: just stare off into space.
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool): pass

# Class for untrained agent which takes random action every step. 
# This class is used as a benchmark at the start of the assignment.
class RandomAgent(RLAlgorithm):
    def __init__(self, actions: List[ActionT]):
        self.actions = actions
    
    def getAction(self, state: StateT, explore: bool = False):
        return random.choice(self.actions)
    
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool):
        pass

def polynomialFeatureExtractor(
        state: StateT,
        degree: int = 3,
        scale: Optional[Iterable] = None
    ) -> np.ndarray:
    '''
    For state (x, y, z), degree 2, and scale [2, 1, 1], this should output:
    [1, 2x, y, z, 4x^2, y^2, z^2, 2xy, 2xz, yz, 4x^2y, 4x^2z, ..., 4x^2y^2z^2]
    '''
    if scale is None:
        scale = np.ones_like(state)
    
    # Create [1, s[0], s[0]^2, ..., s[0]^(degree)] array of shape (degree+1,)
    firstPolyFeat = (state[0] * scale[0])**(np.arange(degree + 1))
    currPolyFeat = firstPolyFeat

    for i in range(1, len(state)):
        # Create [1, s[i], s[i]^2, ..., s[i]^(degree)] array of shape (degree+1,)
        newPolyFeat = (state[i] * scale[i])**(np.arange(degree + 1))

        # Do shape (len(currPolyFeat), 1) times shape (1, degree+1) multiplication 
        # to get broadcasted result of shape (len(currPolyFeat), degree+1)
        # Note that this is also known as the vector outer product.
        currPolyFeat = currPolyFeat.reshape((len(currPolyFeat), 1)) * newPolyFeat.reshape((1, degree + 1))
        
        # Flatten to (len(currPolyFeat) * (degree+1),) array for the next iteration or final features.
        currPolyFeat = currPolyFeat.flatten()
    return currPolyFeat


############################################################

# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Return the list of rewards that we get for each trial.
def simulate(mdp: MDP, rl: RLAlgorithm, numTrials=10, train=True, verbose=False, demo=False):
    
    totalRewards = []  # The discounted rewards we get on each trial
    for trial in range(numTrials):
        state = mdp.startState()
        if demo:
            mdp.env.render()
        totalDiscount = 1
        totalReward = 0
        trialLength = 0
        for _ in range(mdp.timeLimit):
            if demo:
                time.sleep(0.05)
            action = rl.getAction(state, explore=train)
            if action is None: 
                break
            nextState, reward, terminal = mdp.transition(action)
            trialLength += 1
            if train:
                rl.incorporateFeedback(state, action, reward, nextState, terminal)
            
            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount
            state = nextState

            if terminal:
                break # We have reached a terminal state

        if verbose:
            print(("Trial %d (totalReward = %s, Length = %s)" % (trial, totalReward, trialLength)))
        totalRewards.append(totalReward)
    return totalRewards


def sample_RL_trajectory(mdp: MDP, rl: RLAlgorithm, train=True) -> List[Any]:
    traj = []
    state = mdp.startState()
    
    while True:
        action = rl.getAction(state, explore=train)
        if action is None: 
            break
        traj.append(action)
        nextState, reward, terminal = mdp.transition(action)
        if train:
            rl.incorporateFeedback(state, action, reward, nextState, terminal)
        state = nextState

        if terminal:
            break # We have reached a terminal state
    return traj

