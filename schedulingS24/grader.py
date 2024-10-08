#!/usr/bin/env python3
"""
Grader for template assignment
Optionally run as grader.py [basic|all] to run a subset of tests
"""

import random

import graderUtil
import util
import collections
import copy
grader = graderUtil.Grader()
submission = grader.load('submission')


############################################################
# Problem 0a, 0b

grader.add_manual_part('0a', 4, description="Define light bulb CSP")
grader.add_manual_part('0b', 6, description="Solving the CSP")

############################################################
# Problem 0c: Simple Chain CSP

def test0c():
    solver = submission.BacktrackingSearch()
    solver.solve(submission.create_chain_csp(4))
    grader.require_is_equal(1, solver.optimalWeight)
    grader.require_is_equal(2, solver.numOptimalAssignments)
    grader.require_is_equal(9, solver.numOperations)

grader.add_basic_part('0c-1-basic', test0c, 2, max_seconds=1,
        description="Basic test for create_chain_csp")

############################################################
# Problem 1a: N-Queens

def test1a_1():
    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(8))
    grader.require_is_equal(1.0, nQueensSolver.optimalWeight)
    grader.require_is_equal(92, nQueensSolver.numOptimalAssignments)
    grader.require_is_equal(2057, nQueensSolver.numOperations)

grader.add_basic_part('1a-1-basic', test1a_1, 2, max_seconds=1,
        description="Basic test for create_nqueens_csp for n=8")

def test1a_2():
    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(3))

grader.add_hidden_part('1a-2-hidden', test1a_2, 2, max_seconds=1,
        description="Test create_nqueens_csp with n=3")

def test1a_3():
    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(4))

    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(7))

grader.add_hidden_part('1a-3-hidden', test1a_3, 1, max_seconds=1,
        description="Test create_nqueens_csp with different n")

############################################################
# Problem 1b: Most constrained variable


def test1b_1():
    mcvSolver = submission.BacktrackingSearch()
    mcvSolver.solve(submission.create_nqueens_csp(8), mcv = True)
    grader.require_is_equal(1.0, mcvSolver.optimalWeight)
    grader.require_is_equal(92, mcvSolver.numOptimalAssignments)
    grader.require_is_equal(1361, mcvSolver.numOperations)

grader.add_basic_part('1b-1-basic', test1b_1, 1, max_seconds=1,
        description="Basic test for MCV with n-queens CSP")

def test1b_2():
    mcvSolver = submission.BacktrackingSearch()
    # We will use our implementation of n-queens csp
    # mcvSolver.solve(our_nqueens_csp(8), mcv = True)

    # Note to students: This test case will not print anything
    # since there is no call to solve() (on your end). The purpose
    # of this test case is to isolate and test the implementation
    # of get_unassigned_variable() by using our implementation of
    # create_nqueens_csp() so that we can reward students who have
    # an incorrect implementation of create_nqueens_csp() but a
    # correct implementation of get_unassigned_variable().

grader.add_hidden_part('1b-2-hidden', test1b_2, 2, max_seconds=1,
        description="Test for MCV with n-queens CSP")

def test1b_3():
    mcvSolver = submission.BacktrackingSearch()
    mcvSolver.solve(util.create_map_coloring_csp(), mcv = True)

grader.add_hidden_part('1b-3-hidden', test1b_3, 2, max_seconds=1,
        description="Test MCV with different CSPs")

############################################################
# Problem 2

def verify_schedule(bulletin, profile, schedule, checkUnits = True):
    """
    Returns true if the schedule satisifies all requirements given by the profile.
    """
    goodSchedule = True
    all_courses_taking = dict((s[1], s[0]) for s in schedule)

    # No course can be taken twice.
    goodSchedule *= len(all_courses_taking) == len(schedule)
    if not goodSchedule:
        print('course repeated')
        return False

    # Each course must be offered in that quarter.
    goodSchedule *= all(bulletin.courses[s[1]].is_offered_in(s[0]) for s in schedule)
    if not goodSchedule:
        print('course not offered')
        return False

    # If specified, only take the course at the requested time.
    for req in profile.requests:
        if len(req.quarters) == 0: continue
        goodSchedule *= all([s[0] in req.quarters for s in schedule if s[1] in req.cids])
    if not goodSchedule:
        print('course taken at wrong time')
        return False

    # If a request has multiple courses, at most one is chosen.
    for req in profile.requests:
        if len(req.cids) == 1: continue
        goodSchedule *= len([s for s in schedule if s[1] in req.cids]) <= 1
    if not goodSchedule:
        print('more than one exclusive group of courses is taken')
        return False

    # Must take a course after the prereqs
    for req in profile.requests:
        if len(req.prereqs) == 0: continue
        cids = [s for s in schedule if s[1] in req.cids] # either empty or 1 element
        if len(cids) == 0: continue
        quarter, cid, units = cids[0]
        for prereq in req.prereqs:
            if prereq in profile.taking:
                goodSchedule *= prereq in all_courses_taking
                if not goodSchedule:
                    print('not all prereqs are taken')
                    return False
                goodSchedule *= profile.quarters.index(quarter) > \
                    profile.quarters.index(all_courses_taking[prereq])
    if not goodSchedule:
        print('course is taken before prereq')
        return False

    if not checkUnits: return goodSchedule
    # Check for unit loads
    unitCounters = collections.Counter()
    for quarter, c, units in schedule:
        unitCounters[quarter] += units
    goodSchedule *= all(profile.minUnits <= u and u <= profile.maxUnits \
        for k, u in list(unitCounters.items()))
    if not goodSchedule:
        print('unit count out of bound for quarter')
        return False

    return goodSchedule

# Load all courses.
bulletin = util.CourseBulletin('courses.json')

############################################################
# Problem 2a: Quarter specification

def test2a_1():
    profile = util.Profile(bulletin, 'profile2a.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_quarter_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    grader.require_is_equal(3, alg.numOptimalAssignments)
    sol = util.extract_course_scheduling_solution(profile, alg.optimalAssignment)
    for assignment in alg.allAssignments:
        sol = util.extract_course_scheduling_solution(profile, assignment)
        grader.require_is_true(verify_schedule(bulletin, profile, sol, False))

grader.add_basic_part('2a-1-basic', test2a_1, 2, max_seconds=4,
        description="Basic test for add_quarter_constraints")

def test2a_2():
    profile = util.Profile(bulletin, 'profile2a1.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_quarter_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.

grader.add_hidden_part('2a-2-hidden', test2a_2, 2, max_seconds=3,
        description="Test add_quarter_constraints with different profiles")

def test2a_3():
    profile = util.Profile(bulletin, 'profile2a2.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_quarter_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.

grader.add_hidden_part('2a-3-hidden', test2a_3, 2, max_seconds=3,
        description="Test add_quarter_constraints with no quarter specified")

############################################################
# Problem 2b: Unit load

def test2b_1():
    profile = util.Profile(bulletin, 'profile2b.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_unit_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    grader.require_is_equal(15, alg.numOptimalAssignments)
    for assignment in alg.allAssignments:
        sol = util.extract_course_scheduling_solution(profile, assignment)
        grader.require_is_true(verify_schedule(bulletin, profile, sol))

grader.add_basic_part('2b-1-basic', test2b_1, 3, max_seconds=7,
        description="Basic test for add_unit_constraints")

def test2b_2():
    profile = util.Profile(bulletin, 'profile2b1.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_unit_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.

grader.add_hidden_part('2b-2-hidden', test2b_2, 3, max_seconds=3,
        description="Test add_unit_constraints with different profiles")

def test2b_3():
    profile = util.Profile(bulletin, 'profile2b2.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_all_additional_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.

grader.add_hidden_part('2b-3-hidden', test2b_3, 1, max_seconds=4,
        description="Test satisfiable scheduling")

def test2b_4():
    profile = util.Profile(bulletin, 'profile2b3.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_all_additional_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp, mcv = True, ac3 = True)

    # Verify correctness.

grader.add_hidden_part('2b-4-hidden', test2b_4, 3, max_seconds=25,
        description="Test MVC+AC-3+all additional constraints")

grader.add_manual_part('2c', 2, description="Your own schedule")

############################################################
# Problem 3a

grader.add_manual_part('3a', 2, description="Residency Hours Scheduling")

grader.grade()
