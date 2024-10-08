#!/usr/bin/env python3

import graderUtil

grader = graderUtil.Grader()
submission = grader.load('submission')

FINAL_GRADE = True
SEED = 'testing' # random seed at the beginning of each question for more fairness in grading...
BIG_NEGATIVE = -10000

from game import Agent
from ghostAgents import RandomGhost, DirectionalGhost
import random, math, traceback, sys, os

import pacman, time, layout, textDisplay
textDisplay.SLEEP_TIME = 0
textDisplay.DRAW_EVERY = 1000
thismodule = sys.modules[__name__]


def run(layname, pac, ghosts, nGames = 1, name = 'games'):
  """
  Runs a few games and outputs their statistics.
  """
  if grader.fatalError:
    return {'time': 65536, 'wins': 0, 'games': None, 'scores': [0]*nGames, 'timeouts': nGames}

  starttime = time.time()
  lay = layout.getLayout(layname, 3)
  disp = textDisplay.NullGraphics()

  print(('*** Running %s on' % name, layname,'%d time(s).' % nGames))
  games = pacman.runGames(lay, pac, ghosts, disp, nGames, False, catchExceptions=False)
  print(('*** Finished running %s on' % name, layname,'after %d seconds.' % (time.time() - starttime)))

  stats = {'time': time.time() - starttime, 'wins': [g.state.isWin() for g in games].count(True), 'games': games, 'scores': [g.state.getScore() for g in games], 'timeouts': [g.agentTimeout for g in games].count(True)}
  print(('*** Won %d out of %d games. Average score: %f ***' % (stats['wins'], len(games), sum(stats['scores']) * 1.0 / len(games))))

  return stats


def comparison_checking(theirPac, ourPacOptions, agentName):
  """
  Skeleton used for question 2, 3 and 4...
  Takes in their Pacman agent, wraps it in ours, and assigns points.
  """
  print('Running our grader (hidden from you)...')
  random.seed(SEED)
  offByOne = False
  partialPlyBug = False
  totalSuboptimal = 0
  timeout = False


  return timeout, offByOne, partialPlyBug, totalSuboptimal

def test0(agentName):
  stats = {}
  if agentName == 'alphabeta':
    stats = run('smallClassic', submission.AlphaBetaAgent(depth=2), [DirectionalGhost(i + 1) for i in range(2)], name='%s (depth %d)' % ('alphabeta', 2))
  elif agentName == 'minimax':
    stats = run('smallClassic', submission.MinimaxAgent(depth=2), [DirectionalGhost(i + 1) for i in range(2)], name='%s (depth %d)' % ('minimax', 2))
  else:
    stats = run('smallClassic', submission.ExpectimaxAgent(depth=2), [DirectionalGhost(i + 1) for i in range(2)], name='%s (depth %d)' % ('expectimax', 2))
  if stats['timeouts'] > 0:
    grader.fail('Your ' + agentName + ' agent timed out on smallClassic.  No autograder feedback will be provided.')
    return
  grader.assign_full_credit()


gamePlay = {}

def test1(agentName, basic=False):
  if agentName not in gamePlay and not grader.fatalError:
    if agentName == 'minimax':
      gamePlay[agentName] = comparison_checking(submission.MinimaxAgent(depth=2), {}, agentName)
    elif agentName == 'alphabeta':
      gamePlay[agentName] =  comparison_checking(submission.AlphaBetaAgent(depth=2), {agentName: 'True'}, agentName)
    elif agentName == 'expectimax':
      gamePlay[agentName] = comparison_checking(submission.ExpectimaxAgent(depth=2), {agentName: 'True'}, agentName)
    else:
      raise Exception("Unexpected agent name: " + agentName)

  timeout, offByOne, partialPlyBug, totalSuboptimal = gamePlay[agentName]
  if timeout:
    grader.fail('Your ' + agentName + ' agent timed out on smallClassic.  No autograder feedback will be provided.')
    return
    
  if not basic and offByOne:
    grader.fail('Depth off by 1')
  grader.assign_full_credit()

def test2(agentName, basic=False):
  if agentName not in gamePlay and not grader.fatalError:
    if agentName == 'minimax':
      gamePlay[agentName] = comparison_checking(submission.MinimaxAgent(depth=2), {}, agentName)
    elif agentName == 'alphabeta':
      gamePlay[agentName] =  comparison_checking(submission.AlphaBetaAgent(depth=2), {agentName: 'True'}, agentName)
    elif agentName == 'expectimax':
      gamePlay[agentName] = comparison_checking(submission.ExpectimaxAgent(depth=2), {agentName: 'True'}, agentName)
    else:
        raise Exception("Unexpected agent name: " + agentName)

  timeout, offByOne, partialPlyBug, totalSuboptimal = gamePlay[agentName]
  if timeout:
    grader.fail('Your ' + agentName + ' agent timed out on smallClassic.  No autograder feedback will be provided.')
    return
  if not basic and partialPlyBug:
    grader.fail('Incomplete final search ply bug')
  grader.assign_full_credit()

def test3(agentName, basic=False):
  if agentName not in gamePlay and not grader.fatalError:
    if agentName == 'minimax':
      gamePlay[agentName] = comparison_checking(submission.MinimaxAgent(depth=2), {}, agentName)
    elif agentName == 'alphabeta':
      gamePlay[agentName] =  comparison_checking(submission.AlphaBetaAgent(depth=2), {agentName: 'True'}, agentName)
    elif agentName == 'expectimax':
      gamePlay[agentName] = comparison_checking(submission.ExpectimaxAgent(depth=2), {agentName: 'True'}, agentName)
    else:
      raise Exception("Unexpected agent name: " + agentName)

  timeout, offByOne, partialPlyBug, totalSuboptimal = gamePlay[agentName]
  if timeout:
    grader.fail('Your '+agentName+' agent timed out on smallClassic.  No autograder feedback will be provided.')
    return
  if not basic and totalSuboptimal > 0:
    grader.fail('Suboptimal moves: ' + str(totalSuboptimal))
  grader.assign_full_credit()

max_seconds = 10

grader.add_manual_part('1a', 5, description='Recurrence for multi-agent minimiax')

# 1b
grader.add_basic_part('1b-0-basic', lambda : test0('minimax'), 4, max_seconds=max_seconds, description='Tests minimax for timeout on smallClassic.')
grader.add_basic_part('1b-1-basic', lambda : test1('minimax', True), 0, max_seconds=max_seconds, description='Tests minimax for timeout on hidden test 1.')
grader.add_basic_part('1b-2-basic', lambda : test2('minimax', True), 0, max_seconds=max_seconds, description='Tests minimax for timeout on hidden test 2.')
grader.add_basic_part('1b-3-basic', lambda : test3('minimax', True), 0, max_seconds=max_seconds, description='Tests minimax for timeout on hidden test 3.')

grader.add_hidden_part('1b-1-hidden', lambda : test1('minimax'), 2, max_seconds=max_seconds, description='Tests minimax for off by one bug on smallClassic.')
grader.add_hidden_part('1b-2-hidden', lambda : test2('minimax'), 2, max_seconds=max_seconds, description='Tests minimax for search depth bug on smallClassic.')
grader.add_hidden_part('1b-3-hidden', lambda : test3('minimax'), 2, max_seconds=max_seconds, description='Tests minimax for suboptimal moves on smallClassic.')

# 2a
grader.add_basic_part('2a-0-basic', lambda : test0('alphabeta'), 4, max_seconds=max_seconds, description='Tests alphabeta for timeout on smallClassic.')
grader.add_basic_part('2a-1-basic', lambda : test1('alphabeta', True), 0, max_seconds=max_seconds, description='Tests alphabeta for timeout on hidden test 1.')
grader.add_basic_part('2a-2-basic', lambda : test2('alphabeta', True), 0, max_seconds=max_seconds, description='Tests alphabeta for timeout on hidden test 2.')
grader.add_basic_part('2a-3-basic', lambda : test3('alphabeta', True), 0, max_seconds=max_seconds, description='Tests alphabeta for timeout on hidden test 3.')

grader.add_hidden_part('2a-1-hidden', lambda : test1('alphabeta'), 2, max_seconds=max_seconds, description='Tests alphabeta for off by one bug on smallClassic.')
grader.add_hidden_part('2a-2-hidden', lambda : test2('alphabeta'), 2, max_seconds=max_seconds, description='Tests alphabeta for search depth bug on smallClassic.')
grader.add_hidden_part('2a-3-hidden', lambda : test3('alphabeta'), 2, max_seconds=max_seconds, description='Tests alphabeta for suboptimal moves on smallClassic.')

grader.add_manual_part('3a', 5, description='Recurrence for multi-agent expectimax')

# 3b
grader.add_basic_part('3b-0-basic', lambda : test0('expectimax'), 4, max_seconds=max_seconds, description='Tests expectimax for timeout on smallClassic.')
grader.add_basic_part('3b-1-basic', lambda : test1('expectimax', True), 0, max_seconds=max_seconds, description='Tests expectimax for timeout on hidden test 1.')
grader.add_basic_part('3b-2-basic', lambda : test2('expectimax', True), 0, max_seconds=max_seconds, description='Tests expectimax for timeout on hidden test 2.')
grader.add_basic_part('3b-3-basic', lambda : test3('expectimax', True), 0, max_seconds=max_seconds, description='Tests expectimax for timeout on hidden test 3.')

grader.add_hidden_part('3b-1-hidden', lambda : test1('expectimax'), 2, max_seconds=max_seconds, description='Tests expectimax for off by one bug on smallClassic.')
grader.add_hidden_part('3b-2-hidden', lambda : test2('expectimax'), 2, max_seconds=max_seconds, description='Tests expectimax for search depth bug on smallClassic.')
grader.add_hidden_part('3b-3-hidden', lambda : test3('expectimax'), 2, max_seconds=max_seconds, description='Tests expectimax for suboptimal moves on smallClassic.')

############################################################
# Problem 4: evaluation function

def average(list):
  sum = 0.0
  count = 0.0
  for item in list:
    if item is not None:
      sum += item
      count += 1.0
  return 0 if count == 0 else sum / count

def runq4():
  """
  Runs their expectimax agent a few times and checks for victory!
  """
  random.seed(SEED)
  nGames = 20

  print(('Running your agent %d times to compute the average score...' % nGames))
  print(('The timeout message (if any) is obtained by running the game once, rather than %d times' % nGames))
  params = '-l smallClassic -p ExpectimaxAgent -a evalFn=better -q -n %d -c' % nGames
  games = pacman.runGames(**pacman.readCommand(params.split(' ')))
  timeouts = [game.agentTimeout for game in games].count(True)
  wins = [game.state.isWin() for game in games].count(True)
  averageWinScore = 0
  if wins >= nGames / 2:
    averageWinScore = average([game.state.getScore() if game.state.isWin() else None for game in games])
  print(('Average score of winning games: %d \n' % averageWinScore))
  return timeouts, wins, averageWinScore

def testq4():
  timeouts, wins, averageWinScore = 1024, 0, 0

  if not grader.fatalError:
      timeouts, wins, averageWinScore = runq4()

  if timeouts > 0:
    grader.fail('Agent timed out on smallClassic with betterEvaluationFunction. No autograder feedback will be provided.')
    return
  if wins == 0:
    grader.fail('Your better evaluation function never won any games.')
    return
  for score in range(1300, 1800, 100):
    if averageWinScore >= score: grader.add_points(1)

  grader.set_side({'score': averageWinScore})


# EXTRA CREDIT

grader.add_manual_part('4a', 5, extra_credit=True, description='Points for placing in the top 3 (1st place: 5, 2nd place: 3, 3rd place: 1)')
grader.add_basic_part('4a-1-basic', lambda : testq4(), 5, max_seconds=max_seconds, extra_credit=True, description='1 extra credit point per 100 point increase above 1200.')
grader.add_manual_part('4b', 2, extra_credit=True, description='Description of your evaluation function.')

grader.add_manual_part('5a', 2, description='Description of why minimax and expectimax agents differ.')
grader.add_manual_part('5b', 1, description='Suggested change to default state evaluation function.')
grader.add_manual_part('5c', 2, description='Another concrete example of an AI misalignment problem.')

grader.grade()

############################################################
# check python version

import sys
import warnings

if not (sys.version_info[0]==3 and sys.version_info[1]==12):
  warnings.warn(f"Note that you are not using python 3.12. Your code may not work in gradescope.")