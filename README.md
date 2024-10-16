I assigned ClaudeAI to take [Stanford CS221 - Artificial Intelligence: Principles and Techniques](https://stanford-cs221.github.io/autumn2024/). It passed with (mostly) flying colors.
This repository contains code for the completed assignments. Two assignments, foundations and sentiment, are from the current (Fall '24) quarter while the rest are from Spring '24 as the current quarter assignments have not yet been released.

ClaudeAI solved all of the problems in a 24-hour period.

### Foundations
This module consists of some bog-standard warmup problems. Topics include sparse vector operations, a word counting problem, euclidean distance, a rather odd sentence generator, and some alphabetical ordering.

### Sentiment
This module has some simple machine learning. Claude constructed a linear regressor with a hinge loss and spare vectors for sentiment classificaton. Claude also constructed a k-means clustering algorithm for the sparse input vectors. This iss one of the few areas Claude struggled, not with getting a correct k-means algorithm but with getting a fast one. Efficient distance computations required some preprocessing, which Claude could do after being prompted, and finding a clever way to take the dot product over the intersection of two sets of keys, which stumped Claude.

### Route
This module is an application of Dijkstra's algorithm and it's clever enhancement for maps, the A* algorithm. The assignment isn't that hard. Most of the boilerplate code has already been written and all that's left is implementing the cost functions and how to handle start states, end states, and in one problem intermediate states that have to be traversed. Claude handled all of these without issue, but did develop a habit of picking imaginary locations to test. Since Claude is an AI, we call these hallucinations. For anyone else it would be "You lost, buddy?"

### Mountain Car
This module enters the world of reinforcement learning. To get a train car up mountain tracks Claude applied Markov Decision Processses, Value Iteration, and Q-Learning. All tests passed on the first try.

### Pacman
This module concerns game theory, mainly as a subset of reinforcement learning. Minimax, alpha-beta pruning, and Expectimax algorithms are implemented.

### Scheduling
This module solves the course scheduling or constraint satisfaction problem (both abbreviated CSP). Claude had no trouble here.

### Logic
This module evaluates some basic propositional logic. Claude did okay here. It got a couple symbols reversed in one logic statement (So did I; debugging took a little while) and then completely flubbed the extra credit. Even so, I'm confident that Claude outperformed many of its fellow students. 

