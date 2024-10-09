import collections
import os
import sys
from typing import List, Tuple

from logic import *

############################################################
# Problem 1: propositional logic
# Convert each of the following natural language sentences into a propositional
# logic formula.  See rainWet() in examples.py for a relevant example.

# Sentence: "If it's summer and we're in California, then it doesn't rain."
def formula1a() -> Formula:
    Summer = Atom('Summer')
    California = Atom('California')
    Rain = Atom('Rain')
    return Implies(And(Summer, California), Not(Rain))

# Sentence: "It's wet if and only if it is raining or the sprinklers are on."
def formula1b() -> Formula:
    Rain = Atom('Rain')
    Wet = Atom('Wet')
    Sprinklers = Atom('Sprinklers')
    return Equiv(Wet, Or(Rain, Sprinklers))

# Sentence: "Either it's day or night (but not both)."
def formula1c() -> Formula:
    Day = Atom('Day')
    Night = Atom('Night')
    return Xor(Day, Night)

############################################################
# Problem 2: first-order logic

# Sentence: "Every person has a parent."
def formula2a() -> Formula:
    def Person(x): return Atom('Person', x)
    def Parent(x, y): return Atom('Parent', x, y)
    return Forall('$x', Implies(Person('$x'), Exists('$y', Parent('$x', '$y'))))

# Sentence: "At least one person has no children."
def formula2b() -> Formula:
    def Person(x): return Atom('Person', x)
    def Child(x, y): return Atom('Child', x, y)
    return Exists('$x', And(Person('$x'), Not(Exists('$y', Child('$x', '$y')))))

# Return a formula which defines Father in terms of Male and Parent
# See parentChild() in examples.py for a relevant example.
def formula2c() -> Formula:
    def Male(x): return Atom('Male', x)
    def Parent(x, y): return Atom('Parent', x, y)
    def Father(x, y): return Atom('Father', x, y)

    def MaleParent(x,y): return And(Male(y), Parent(x, y))
    return Forall('$x', Forall('$y', Equiv(MaleParent('$x','$y'), Father('$x', '$y'))))

# Return a formula which defines Granddaughter in terms of Female and Child.
# Note: It is ok for a person to be her own child
def formula2d() -> Formula:
    def Female(x): return Atom('Female', x)
    def Child(x, y): return Atom('Child', x, y)
    def Granddaughter(x, y): return Atom('Granddaughter', x, y)
    return Forall('$x', Forall('$y', Equiv(Granddaughter('$x', '$y'),
                                           And(Female('$y'),
                                               Exists('$z', And(Child('$x', '$z'), Child('$z', '$y')))))))

############################################################
# Problem 3: Liar puzzle

# Facts:
# 0. Mark: "It wasn't me!"
# 1. John: "It was Nicole!"
# 2. Nicole: "No, it was Susan!"
# 3. Susan: "Nicole's a liar."
# 4. Exactly one person is telling the truth.
# 5. Exactly one person crashed the server.
# Query: Who did it?
# This function returns a list of 6 formulas corresponding to each of the
# above facts. Be sure your formulas are exactly in the order specified. 
# Hint: You might want to use the Equals predicate, defined in logic.py.  This
# predicate is used to assert that two objects are the same.
# In particular, Equals(x,x) = True and Equals(x,y) = False iff x is not equal to y.
def liar() -> Tuple[List[Formula], Formula]:
    def TellTruth(x): return Atom('TellTruth', x)
    def CrashedServer(x): return Atom('CrashedServer', x)
    mark = Constant('mark')
    john = Constant('john')
    nicole = Constant('nicole')
    susan = Constant('susan')
    formulas = []
    # Fact 0
    formulas.append(Equiv(TellTruth(mark), Not(CrashedServer(mark))))
    # Fact 1
    formulas.append(Equiv(TellTruth(john), CrashedServer(nicole)))
    # Fact 2
    formulas.append(Equiv(TellTruth(nicole), CrashedServer(susan)))
    # Fact 3
    formulas.append(Equiv(TellTruth(susan), Not(TellTruth(nicole))))
    # Fact 4
    formulas.append(Exists('$x', And(TellTruth('$x'),
                    Forall('$y', Implies(Not(Equals('$x', '$y')), Not(TellTruth('$y')))))))
    # Fact 5
    formulas.append(Exists('$x', And(CrashedServer('$x'),
                    Forall('$y', Implies(Not(Equals('$x', '$y')), Not(CrashedServer('$y')))))))
    query = CrashedServer('$x')
    return (formulas, query)

############################################################
# Problem 4: Odd and even integers

# Return the following 6 laws. Be sure your formulas are exactly in the order specified.
# 0. Each number $x$ has exactly one successor, which is not equal to $x$.
# 1. Each number is either even or odd, but not both.
# 2. The successor number of an even number is odd.
# 3. The successor number of an odd number is even.
# 4. For every number $x$, the successor of $x$ is larger than $x$.
# 5. Larger is a transitive property: if $x$ is larger than $y$ and $y$ is
#    larger than $z$, then $x$ is larger than $z$.
# Query: For each number, there exists an even number larger than it.
def ints() -> Tuple[List[Formula], Formula]:
    def Even(x): return Atom('Even', x)
    def Odd(x): return Atom('Odd', x)
    def Successor(x, y): return Atom('Successor', x, y)
    def Larger(x, y): return Atom('Larger', x, y)

    formulas = []
    # Law 0
    formulas.append(Forall('$x', Exists('$y', And(And(Successor('$x', '$y'),
                                                      Not(Equals('$x', '$y'))),
                                                  Forall('$z', Implies(Successor('$x', '$z'), Equals('$y', '$z')))))))
    # Law 1
    formulas.append(Forall('$x', Xor(Even('$x'), Odd('$x'))))
    # Law 2
    formulas.append(Forall('$x', Forall('$y', Implies(And(Even('$x'), Successor('$x', '$y')), Odd('$y')))))
    # Law 3
    formulas.append(Forall('$x', Forall('$y', Implies(And(Odd('$x'), Successor('$x', '$y')), Even('$y')))))
    # Law 4
    formulas.append(Forall('$x', Forall('$y', Implies(Successor('$x', '$y'), Larger('$y', '$x')))))
    # Law 5
    formulas.append(Forall('$x', Forall('$y', Forall('$z', Implies(And(Larger('$x', '$y'), Larger('$y', '$z')),
                                                                   Larger('$x', '$z'))))))

    query = Forall('$x', Exists('$y', And(Even('$y'), Larger('$y', '$x'))))
    return (formulas, query)

############################################################
# Problem 5: semantic parsing
# Each of the following functions should return a GrammarRule.
# Look at createBaseEnglishGrammar() in nlparser.py to see what these rules should look like.
# For example, the rule for 'X is a Y' is:
#     GrammarRule('$Clause', ['$Name', 'is', 'a', '$Noun'],
#                 lambda args: Atom(args[1].title(), args[0].lower()))
# Note: args[0] corresponds to $Name and args[1] corresponds to $Noun.
# Note: by convention, .title() should be applied to all predicates (e.g., Cat).
# Note: by convention, .lower() should be applied to constant symbols (e.g., garfield).

from nlparser import GrammarRule


def createRule1() -> GrammarRule:
    return GrammarRule('$Clause', ['every', '$Noun', '$Verb', 'some', '$Noun'],
        lambda args: Forall('$x', Implies(Atom(args[0].title(), '$x'),
                            Exists('$y', And(Atom(args[2].title(), '$y'),
                                             Atom(args[1].title(), '$x', '$y'))))))

def createRule2() -> GrammarRule:
    return GrammarRule('$Clause', ['there', 'is', 'some', '$Noun', 'that', 'every', '$Noun', '$Verb'],
        lambda args: Exists('$x', And(Atom(args[0].title(), '$x'),
                            Forall('$y', Implies(Atom(args[1].title(), '$y'),
                                                 Atom(args[2].title(), '$y', '$x'))))))

def createRule3() -> GrammarRule:
    return GrammarRule('$Clause', ['if', 'a', '$Noun', '$Verb', 'a', '$Noun', 'then', 'the', 'former', '$Verb', 'the', 'latter'],
        lambda args: Forall('$x', Forall('$y', Implies(And(Atom(args[0].title(), '$x'),
                                                           Atom(args[1].title(), '$x', '$y'),
                                                           Atom(args[2].title(), '$y')),
                                                       Atom(args[3].title(), '$x', '$y')))))
