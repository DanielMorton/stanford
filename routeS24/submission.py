from typing import List, Tuple

from mapUtil import (
    CityMap,
    computeDistance,
    createStanfordMap,
    locationFromTag,
    makeTag,
)
from util import Heuristic, SearchProblem, State, UniformCostSearch


# *IMPORTANT* :: A key part of this assignment is figuring out how to model states
# effectively. We've defined a class `State` to help you think through this, with a
# field called `memory`.
#
# As you implement the different types of search problems below, think about what
# `memory` should contain to enable efficient search!
#   > Please read the docstring for `State` in `util.py` for more details and code.
#   > Please read the docstrings for in `mapUtil.py`, especially for the CityMap class

########################################################################################
# Problem 1a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `startLocation` to any location with the specified `endTag`.
    """

    def __init__(self, startLocation: str, endTag: str, cityMap: CityMap):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

    def startState(self) -> State:
        return State(location=self.startLocation)

    def isEnd(self, state: State) -> bool:
        return self.endTag in self.cityMap.tags[state.location]

    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        """
        Note we want to return a list of *3-tuples* of the form:
            (actionToReachSuccessor: str, successorState: State, cost: float)
        """
        successors = []
        for neighbor, distance in self.cityMap.distances[state.location].items():
            successors.append((neighbor, State(location=neighbor), distance))
        return successors


########################################################################################
# Problem 1b: Custom -- Plan a Route through Stanford


def getStanfordShortestPathProblem() -> ShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`endTag`. 

    Run `python mapUtil.py > readableStanfordMap.txt` to dump a file with a list of
    locations and associated tags; you might find it useful to search for the following
    tag keys (amongst others):
        - `landmark=` - Hand-defined landmarks (from `data/stanford-landmarks.json`)
        - `amenity=`  - Various amenity types (e.g., "parking_entrance", "food")
        - `parking=`  - Assorted parking options (e.g., "underground")
    """
    cityMap = createStanfordMap()

    startLocation = "314071068"  # Location ID for the Main Quad
    endTag = "landmark=bookstore"  # Tag for Hoover Tower

    return ShortestPathProblem(startLocation, endTag, cityMap)


########################################################################################
# Problem 2a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`.
    """
    def __init__(
        self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap
    ):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap
        self.waypointTags = tuple(sorted(waypointTags))

    def startState(self) -> State:
        initial_waypoints = tuple(
            tag for tag in self.waypointTags
            if tag in self.cityMap.tags[self.startLocation]
        )
        return State(location=self.startLocation, memory=initial_waypoints)

    def isEnd(self, state: State) -> bool:
        all_waypoints_visited = set(state.memory) == set(self.waypointTags)
        end_tag_present = self.endTag in self.cityMap.tags[state.location]
        return all_waypoints_visited and end_tag_present

    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        successors = []
        for neighbor, distance in self.cityMap.distances[state.location].items():
            new_waypoints = set(state.memory)
            for tag in self.waypointTags:
                if tag in self.cityMap.tags[neighbor]:
                    new_waypoints.add(tag)
            new_state = State(location=neighbor, memory=tuple(sorted(new_waypoints)))
            successors.append((neighbor, new_state, distance))
        return successors


########################################################################################
# Problem 2c: Custom -- Plan a Route with Unordered Waypoints through Stanford


def getStanfordWaypointsShortestPathProblem() -> WaypointsShortestPathProblem:
    """
    Create your own search problem with waypoints using the map of Stanford, 
    specifying your own `startLocation`/`waypointTags`/`endTag`.

    Similar to Problem 1b, use `readableStanfordMap.txt` to identify potential
    locations and tags.
    """
    cityMap = createStanfordMap()

    startLocation = "314071068"  # Location ID for the Main Quad
    waypointTags = [makeTag("landmark", "hoover_tower"),
                    makeTag("landmark", "bookstore"),
                    makeTag("landmark", 'oval')]
    endTag = "landmark=hoover_tower"

    return WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)


########################################################################################
# Problem 3a: A* to UCS reduction

# Turn an existing SearchProblem (`problem`) you are trying to solve with a
# Heuristic (`heuristic`) into a new SearchProblem (`newSearchProblem`), such
# that running uniform cost search on `newSearchProblem` is equivalent to
# running A* on `problem` subject to `heuristic`.
#
# This process of translating a model of a problem + extra constraints into a
# new instance of the same problem is called a reduction; it's a powerful tool
# for writing down "new" models in a language we're already familiar with.


def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def startState(self) -> State:
            return problem.startState()

        def isEnd(self, state: State) -> bool:
            return problem.isEnd(state)

        def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            successors = []
            for action, newState, cost in problem.actionSuccessorsAndCosts(state):
                # Calculate f(n) = g(n) + h(n)
                # g(n) is the cost to reach newState
                # h(n) is the heuristic estimate from newState to the goal
                f_cost = cost + heuristic.evaluate(newState) - heuristic.evaluate(state)
                successors.append((action, newState, f_cost))
            return successors

    return NewSearchProblem()


########################################################################################
# Problem 3b: "straight-line" heuristic for A*


class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

        # Precompute end locations
        self.endLocations = []
        for location, tags in self.cityMap.tags.items():
            if self.endTag in tags:
                self.endLocations.append(location)

    def evaluate(self, state: State) -> float:
        """
        Compute the minimum straight-line distance from the current state
        to any of the end locations.
        """
        current_location = self.cityMap.geoLocations[state.location]
        min_distance = float('inf')

        for end_location in self.endLocations:
            end_geo = self.cityMap.geoLocations[end_location]
            distance = computeDistance(current_location, end_geo)
            min_distance = min(min_distance, distance)

        return min_distance


########################################################################################
# Problem 3c: "no waypoints" heuristic for A*


class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `startLocation` to any location with `endTag`,
    ignoring all waypoints.
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        """
        Precompute cost of shortest path from each location to a location with the desired endTag
        """
        class ReverseShortestPathProblem(SearchProblem):
            def __init__(self, endTag, cityMap):
                self.endTag = endTag
                self.cityMap = cityMap

            def startState(self) -> State:
                """
                Return special "END" state
                """
                return State("END")

            def isEnd(self, state: State) -> bool:
                """
                Return False for each state.
                Because there is *not* a valid end state (`isEnd` always returns False),
                UCS will exhaustively compute costs to *all* other states.
                """
                return False

            def actionSuccessorsAndCosts(
                self, state: State
            ) -> List[Tuple[str, State, float]]:
                if state.location == "END":
                    return [(loc, State(loc), 0) for loc, tags in self.cityMap.tags.items() if self.endTag in tags]
                else:
                    return [(neigh, State(neigh), cost)
                            for neigh, cost in self.cityMap.distances[state.location].items()]

        # Call UCS.solve on our `ReverseShortestPathProblem` instance
        ucs = UniformCostSearch()
        ucs.solve(ReverseShortestPathProblem(endTag, cityMap))

        # Store the precomputed costs
        self.costs = ucs.pastCosts

    def evaluate(self, state: State) -> float:
        return self.costs.get(state.location, float('inf'))
