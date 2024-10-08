#!/usr/bin/env python3

import random, util, collections, json, math
import graderUtil
import numpy as np

grader = graderUtil.Grader()
submission = grader.load('submission')

############################################################
# Problem 1

grader.add_manual_part('1a', 3, description="Written question: value iteration in basic MDP")
grader.add_manual_part('1b', 1, description="Written question: optimal policy in basic MDP")

############################################################
# Problem 2

grader.add_manual_part('2a', 4, description="Written question: define new MDP solver for discounts < 1")


############################################################
# Problem 3

def test3a0():
    mdp = util.NumberLineMDP()
    pi = submission.run_VI_over_numberLine(mdp)
    gold = {
        -1: 1,
        1: 2,
        0: 2
    }
    for key in pi:
        if not grader.require_is_equal(pi[key], gold[key]):
            print("Incorrect pi for the state:", key)
grader.add_basic_part('3a-0-basic', test3a0, 1, description="Basic test of VI on problem 1.")

def test3a1():
    mdp = util.NumberLineMDP(10, 30, -1, 20)
    pi = submission.run_VI_over_numberLine(mdp)
    with open("3a-1-gold.json", "r") as f:
        gold = json.load(f)
    for key in gold:
        key_i = int(key)
        if key_i not in pi:
            msg = "Optimal policy for state " + key + " not computed!"
            grader.fail(msg)
        elif not grader.require_is_equal(pi[key_i], gold[key]):
            print("Incorrect pi for the state:", key)

grader.add_basic_part('3a-1-basic', test3a1, 2, description="Test on arbitrary n, reward and penalty.")

def test3a2Hidden():
    mdp = util.NumberLineMDP(n=500)
    pi = submission.run_VI_over_numberLine(mdp)

grader.add_hidden_part('3a-2-hidden', test3a2Hidden, max_points=2, max_seconds=14, description="Hidden test to make sure the code runs fast enough")

def test3b0():
    mdp = util.NumberLineMDP()
    rl = submission.ModelBasedMonteCarlo(mdp.actions, mdp.discount, calcValIterEvery=1, explorationProb=0.2)
    rl.pi = {
        -1: 1,
        1: 2,
        0: 2
    }
    rl.numIters = 2e4
    counts = {
        -1: 0,
        0: 0,
        1: 0
    }
    for _ in range(10000):
        for state in range(-mdp.n + 1, mdp.n):
            action = rl.getAction(state)
            if action == rl.pi[state]:
                counts[state] += 1
    for key in counts:
        if not grader.require_is_greater_than(8800, counts[key]):
            print("Too few optimal actions returned for the state", key)
        if not grader.require_is_less_than(9200, counts[key]):
            print("Too few random actions returned for the state", key)

grader.add_basic_part('3b-0-basic', test3b0, max_points=2, description="testing epsilon greedy for getAction.")

def test3b1():
    mdp = util.NumberLineMDP()
    rl = submission.ModelBasedMonteCarlo(mdp.actions, mdp.discount, calcValIterEvery=100, explorationProb=0.2)
    rl.numIters = 1
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, -5, 0, False)
    rl.numIters = 100
    rl.incorporateFeedback(-1, 2, 10, -2, True)
    gold = {1:1, -1:2}
    if rl.pi != gold:
        msg = "Incorrect implementation of incorporateFeedback!"
        grader.fail(msg)
    else:
        grader.assign_full_credit()

grader.add_basic_part('3b-1-basic', test3b1, max_points=2, description="basic test of incorporate feedback.")

def test3b2():
    mdp = util.NumberLineMDP()
    rl = submission.ModelBasedMonteCarlo(mdp.actions, mdp.discount, calcValIterEvery=100, explorationProb=0.2)
    rl.numIters = 1
    rl.incorporateFeedback(0, 1, -5, 1, False)
    rl.incorporateFeedback(0, 1, -5, 1, False)
    rl.incorporateFeedback(0, 1, -5, -1, False)
    rl.incorporateFeedback(0, 2, -5, 1, False)
    rl.incorporateFeedback(0, 2, -5, -1, False)
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, -5, 0, False)
    rl.incorporateFeedback(1, 2, 50, 2, True)
    rl.incorporateFeedback(1, 2, -5, 0, False)
    rl.incorporateFeedback(-1, 1, -5, 0, False)
    rl.incorporateFeedback(-1, 2, -5, 0, False)
    rl.numIters = 100
    rl.incorporateFeedback(-1, 2, 10, -2, True)
    gold = {
        0: 1,
        1: 1,
        -1: 1
    }
    for key in gold:
        if not grader.require_is_equal(gold[key], rl.pi[key]):
            print("Incorrect pi for the state", key, "after MC value iteration!")

grader.add_basic_part('3b-2-basic', test3b2, max_points=4, description="comprehensive test for incorporateFeedback.")

def test3b3():
    mdp = util.NumberLineMDP()
    rl = submission.ModelBasedMonteCarlo(mdp.actions, mdp.discount, calcValIterEvery=1, explorationProb=0.2)
    counts = dict()
    for state in range(-mdp.n, mdp.n + 1):
        counts[state] = 0
    for _ in range(10000):
        for state in counts:
            action = rl.getAction(state)
            if action == 1:
                counts[state] += 1
    if counts[-mdp.n] < 4000 or counts[mdp.n] > 6000:
        grader.fail("Wrong edge case handling!")
    else:
        grader.assign_full_credit()

grader.add_basic_part('3b-3-basic', test3b3, max_points=2, max_seconds=5, description="Edge case handling.")

grader.add_manual_part('3c', 2, description="Written question: discussion of MC Value Iteration performance.")

############################################################
# Problem 4

def test4a0():
    mdp = util.NumberLineMDP()
    rl = submission.TabularQLearning(mdp.actions, mdp.discount, explorationProb=0.15)
    rl.incorporateFeedback(0, 1, -5, 1, False)
    grader.require_is_equal(0, rl.Q[(1, 2)])
    grader.require_is_equal(0, rl.Q[(1, 1)])
    grader.require_is_equal(-0.5, rl.Q[(0, 1)])
    rl.incorporateFeedback(1, 1, 50, 2, True)
    grader.require_is_equal(5.0, rl.Q[(1,1)])
    grader.require_is_equal(0, rl.Q[(1,2)])
    grader.require_is_equal(-0.5, rl.Q[(0,1)])
    rl.incorporateFeedback(-1, 2, -5, 0, False)
    grader.require_is_equal(5.0, rl.Q[(1, 1)])
    grader.require_is_equal(0, rl.Q[(1, 2)])
    grader.require_is_equal(-0.5, rl.Q[(0, 1)])
    grader.require_is_equal(0, rl.Q[(0, 2)])
    grader.require_is_equal(-0.5, rl.Q[(-1, 2)])

grader.add_basic_part('4a-0-basic', test4a0, max_points=5, max_seconds=5, description="Basic test for incorporateFeedback.")


def test4a1():
    mdp = util.NumberLineMDP()
    rl = submission.TabularQLearning(mdp.actions, mdp.discount, explorationProb=0.15)
    rl.incorporateFeedback(0, 1, -5, 1, False)
    rl.incorporateFeedback(0, 1, -5, 1, False)
    rl.incorporateFeedback(0, 1, -5, -1, False)
    rl.incorporateFeedback(0, 2, -5, 1, False)
    rl.incorporateFeedback(0, 2, -5, -1, False)
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, -5, 0, False)
    rl.incorporateFeedback(1, 2, 50, 2, True)
    rl.incorporateFeedback(1, 2, -5, 0, False)
    rl.incorporateFeedback(-1, 1, -5, 0, False)
    rl.incorporateFeedback(-1, 1, 10, -2, True)
    rl.incorporateFeedback(-1, 2, -5, 0, False)
    pi = {
        -1: 1,
        0: 2,
        1: 1
    }
    for state in range(-mdp.n+1, mdp.n):
        if not grader.require_is_equal(pi[state], rl.getAction(state, explore=False)):
            print("Incorrect greedy action with the state", state)

grader.add_basic_part('4a-1-basic', test4a1, max_points=3, max_seconds=5, description="Basic test for getAction.")

def test4a2Hidden():
    mdp = util.NumberLineMDP()
    rl = submission.TabularQLearning(mdp.actions, mdp.discount, explorationProb=0.15)
    rl.incorporateFeedback(0, 1, -5, 1, False)
    rl.incorporateFeedback(0, 1, -5, 1, False)
    rl.incorporateFeedback(0, 1, -5, -1, False)
    rl.incorporateFeedback(0, 2, -5, 1, False)
    rl.incorporateFeedback(0, 2, -5, -1, False)
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, -5, 0, False)
    rl.incorporateFeedback(1, 2, 50, 2, True)
    rl.incorporateFeedback(1, 2, -5, 0, False)
    rl.incorporateFeedback(-1, 1, -5, 0, False)
    rl.incorporateFeedback(-1, 1, 10, -2, True)
    rl.incorporateFeedback(-1, 2, -5, 0, False)

grader.add_hidden_part('4a-2-hidden', test4a2Hidden, max_points=2, max_seconds=5, description="Hidden test for getAction.")

def test4b0():
    feature = submission.fourierFeatureExtractor((0.5, 0.3))
    gold = np.load("4b-0-gold.npy", allow_pickle=True)
    if not grader.require_is_equal(gold.size, feature.size):
        print("Returned feature does not have the correct dimension!")
    gold_sorted = np.sort(gold)
    feature_sorted = np.sort(feature)
    for i in range(feature.size):
        if not math.isclose(feature_sorted[i], gold_sorted[i]):
            msg = "Wrong value for an element of the feature: expected " + str(gold_sorted[i]) + " but got " + str(feature_sorted[i])
            grader.fail(msg)

grader.add_basic_part('4b-0-basic', test4b0, max_points=5, description="Basic test of fourierFeatureExtractor.")

def test4c0():
    mdp = util.ContinuousGymMDP("MountainCar-v0", discount=0.999, timeLimit=1000)
    rl = submission.FunctionApproxQLearning(36,
        lambda s: submission.fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
        mdp.actions, mdp.discount, explorationProb=0.2)
    rl.W = np.zeros((36, 3))
    rl.incorporateFeedback((0, 0), 1, -1, (-0.2, -0.01), False)
    rl.incorporateFeedback((0.7, -0.03), 2, -1, (0.8, -0.01), False)
    rl.incorporateFeedback((-0.3, -0.05), 0, -1, (-0.4, -0.03), False)
    if not math.isclose(rl.getQ((0.2, -0.02), 1), -0.0074065262637628875, abs_tol=1e-6):
        msg = "Wrong Q value computed for given state and action!"
        grader.fail(msg)
    else:
        grader.assign_full_credit()

grader.add_basic_part('4c-0-basic', test4c0, max_points=2, description="Basic tests for getQ on FA.")

def test4c1():
    mdp = util.ContinuousGymMDP("MountainCar-v0", discount=0.999, timeLimit=1000)
    rl = submission.FunctionApproxQLearning(36,
        lambda s: submission.fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
        mdp.actions, mdp.discount, explorationProb=0.2)
    rl.W = np.zeros((36, 3))
    rl.incorporateFeedback((0, 0), 1, -1, (-0.2, -0.01), False)
    rl.incorporateFeedback((0.7, -0.03), 2, -1, (0.8, -0.01), False)
    rl.incorporateFeedback((-0.3, -0.05), 0, -1, (-0.4, -0.03), False)
    action = rl.getAction((0.2, -0.02), explore=False)
    if not grader.require_is_equal(0, action):
        print("Wrong action based on current weight!")
    action = rl.getAction((1, 0.03), explore=False)
    if not grader.require_is_equal(2, action):
        print("Wrong action based on current weight!")
    action = rl.getAction((-0.6, -0.06), explore=False)
    if not grader.require_is_equal(0, action):
        print("Wrong action based on current weight!")

grader.add_basic_part('4c-1-basic', test4c1, max_points=3, description="Basic tests for getAction on FA.")

def test4c2():
    mdp = util.ContinuousGymMDP("MountainCar-v0", discount=0.999, timeLimit=1000)
    rl = submission.FunctionApproxQLearning(36,
        lambda s: submission.fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
        mdp.actions, mdp.discount, explorationProb=0.2)
    rl.W = np.zeros((36, 3))
    rl.incorporateFeedback((0, 0), 1, -1, (-0.2, -0.01), False)
    rl.incorporateFeedback((0.7, -0.03), 2, -1, (0.8, -0.01), False)
    rl.incorporateFeedback((-0.3, -0.05), 0, -1, (-0.4, -0.03), False)
    gold = np.load("4c-2-gold.npy", allow_pickle=True)
    for i in range(36):
        for j in range(36):
            if np.all(np.isclose(gold[i], rl.W[j], atol=1e-6)): # good, so break
                break
        else: # no break
            msg = "Weight update incorrect!"
            grader.fail(msg)
            print(msg)
            return
        for j in range(36):
            if np.all(np.isclose(gold[j], rl.W[i], atol=1e-6)): # good, so break
                break
        else: # no break
            msg = "Weight update incorrect!"
            grader.fail(msg)
            print(msg)
            return
    grader.assign_full_credit()

grader.add_basic_part('4c-2-basic', test4c2, max_points=5, description="Basic tests for incorporateFeedback on FA.")

grader.add_manual_part('4d', 2, description="Written question: discussion of Q-Learning performance.")

grader.add_manual_part('4e', 2, description="Written question: Advantages of function approximation Q-Learning.")

############################################################
# Problem 5

grader.add_manual_part('5a', 2, description="Written question: self.max_speed")
grader.add_manual_part('5b', 1, description="Written question: removing max_speed")
grader.add_manual_part('5c', 2, description="Written question: output and reward of constrained QL")

# NOTE: as in 4b above, this is not a real test -- just a helper function to run some code
# to produce stats that will allow you to answer written question 5b.
def run5cHelper():
    submission.compare_MDP_Strategies(submission.mdp1, submission.mdp2)


# grader.add_basic_part('5c-helper', run5cHelper, 0, max_seconds=60,
#                       description="Helper function to compare optimal policies over max speed constraints.")  #
grader.add_manual_part('5d', 2, description="Written question: real world safe RL context")

grader.grade()
