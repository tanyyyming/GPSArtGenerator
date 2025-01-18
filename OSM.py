from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from scipy.spatial.distance import euclidean


def get_roads(bbox: list[float, float, float, float]) -> list:
    """
    The bbox is a list in the form of [min_lat(south), min_lon(west), max_lat(north), max_lon(east)]

    Returns a list of roads in the bbox, each consists of nodes in the form of (node_id, [lat, lon])
    """
    overpass = Overpass()
    query = overpassQueryBuilder(
        bbox=bbox,
        elementType="way",
        # TODO: Refine the road types for better abstraction
        selector='"highway"~"primary|secondary|tertiary|unclassified|residential|pedestrian|service"',
        out="body",
        includeGeometry=True,
    )
    result = overpass.query(query)
    return result.elements()


def construct_graph(roads: list) -> dict[int, list[int]]:
    """
    The roads is a list of the OSM elements, each represents a road.
    Each road is represented by its constituent nodes.
    The function returns an adjacency list representation of the graph.
    """
    graph = {}
    for road in roads:
        # nodes on the road
        nodes = [nd.id() for nd in road.nodes()]
        for nd1, nd2 in zip(nodes[:-1], nodes[1:]):
            if nd1 not in graph:
                graph[nd1] = []
            if nd2 not in graph:
                graph[nd2] = []
            graph[nd1].append(nd2)
            graph[nd2].append(nd1)
    return graph


def construct_node_coordinates(roads: list) -> dict[int, tuple[float, float]]:
    """
    Construct a dictionary of node_id to its coordinates
    """
    node_coordinates = {}
    for road in roads:
        # nodes on the road
        nodes = [nd.id() for nd in road.nodes()]
        # coordinates of each node on the road
        coordinates = road.geometry()["coordinates"]
        type = road.geometry()["type"]
        try:
            assert (type == "LineString" and len(nodes) == len(coordinates)) or (
                # account for the special case where the road is a closed loop
                type == "Polygon"
                and len(nodes) == len(coordinates[0])
            )
            # handle the closed loop case
            if type == "Polygon":
                coordinates = coordinates[0]
            for i in range(len(nodes)):
                # the value of the coordinates is in the form of (lon, lat), so we reverse it
                node_coordinates[nodes[i]] = tuple(coordinates[i][::-1])
        except AssertionError:
            print(f"Way {road.id()} has mismatched nodes and coordinates")
            print("Number of nodes:", len(nodes))
            print("Number of coordinates:", len(coordinates))
    return node_coordinates


def get_closest_node(
    coordinates: tuple[float, float],
    nodes: dict[int, tuple[float, float]],
) -> int:
    """
    Get the closest node to the given coordinates
    """
    best_node = min(
        list(nodes.keys()),
        key=lambda id: euclidean(nodes[id], coordinates),
    )
    return best_node
