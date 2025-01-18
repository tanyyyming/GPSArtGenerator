import numpy as np
from scipy.spatial.distance import euclidean, directed_hausdorff
from heapq import heappop, heappush
from math import inf
from collections import defaultdict
from typing import Optional


def dijkstra(
    graph: dict[int, list[int]],
    coordinates: dict[int, tuple[float, float]],
    start_node: int,
    end_node: int,
) -> Optional[list[tuple[float, float]]]:
    def get_neighbors(node: int) -> list[tuple[int, float]]:
        neighbors = graph[node]
        weights = [
            cost_function(
                coordinates[node],
                coordinates[neighbor],
                coordinates[start_node],
                coordinates[end_node],
            )
            for neighbor in neighbors
        ]
        return list(zip(neighbors, weights))

    def bfs(root: int, target: int) -> tuple[int, dict[int, int]]:
        queue = [(0, root)]
        # distances are in the sense of the modified cost function,
        # not the original euclidean distance.
        distances = defaultdict(lambda: inf)
        parents = defaultdict(lambda: None)
        distances[root] = 0
        parents[root] = root
        while len(queue) > 0:
            distance, node = heappop(queue)
            if node == target:
                return distance, parents
            if distance > distances[node]:
                continue
            for neighbor, weight in get_neighbors(node):
                d = distances[node] + weight
                if distances[neighbor] <= d:
                    continue
                heappush(queue, (d, neighbor))
                distances[neighbor] = d
                parents[neighbor] = node
        return distances[target], parents

    min_dist, parents = bfs(start_node, end_node)

    if min_dist == inf:
        return None
    else:
        path = [end_node]
        while path[-1] != start_node:
            path.append(parents[path[-1]])
        return [coordinates[node] for node in reversed(path)]


def cost_function(
    curr_node_coords: tuple[float, float],
    next_node_coords: tuple[float, float],
    start_node_coords: tuple[float, float],
    end_node_coords: tuple[float, float],
    alpha: float = 0,
    beta: float = 0,
    gamma: float = 1,
) -> float:
    """
    The modified weight/cost function of the Dijkstra algorithm is
    C(P, N, S, E) = (αC1(N, E) + β + γC3(P, N, S, E)) * C2(P, N)
    where α, β, and γ are the weights of the three different subcost functions.
    C2(P, N) is multiplied instead of added to normalize the cost function s.t.
    paths with denser nodes are not disadvantaged.

    Parameters:
    P: the current node
    N: the next node
    S: the start node
    E: the end node
    """

    def distance_to_end_node(N: tuple[float, float], E: tuple[float, float]) -> float:
        """
        Calculate the distance from the next node N to the end node E

        C1(N, E) = |N - E|
        """
        return euclidean(N, E)

    def edge_weight(P: tuple[float, float], N: tuple[float, float]) -> float:
        """
        Calculate the edge weight between the current node P and the next node N.
        This is the original weight function of the Dijkstra algorithm.

        C2(P, N) = |P - N|
        """
        return euclidean(P, N)

    def closeness_to_line_segment(
        P: tuple[float, float],
        N: tuple[float, float],
        S: tuple[float, float],
        E: tuple[float, float],
        num_points: int = 100,
    ) -> float:
        """
        Calculate the closeness of the line PN to the line segment SE we would like to approximate.
        Reducing this value will make the path more likely to follow the line segment and
        thus make a better GPS art.
        Note that since hausdorff is discrete, we have to convert PN and SE each into
        equidistant points in order to apply the SciPy function.

        C3(P, N, S, E) = hausdorff distance between PN and SE
        """
        PN = np.linspace(P, N, num_points)
        SE = np.linspace(S, E, num_points)
        return directed_hausdorff(PN, SE)[0]
        # return max(directed_hausdorff(PN, SE)[0], directed_hausdorff(SE, PN)[0])

    return (
        alpha * distance_to_end_node(next_node_coords, end_node_coords)
        + beta
        + gamma
        * closeness_to_line_segment(
            curr_node_coords, next_node_coords, start_node_coords, end_node_coords
        )
    ) * edge_weight(curr_node_coords, next_node_coords)

