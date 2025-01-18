from OSM import get_roads, construct_graph, construct_node_coordinates, get_closest_node
from shape_processor import line_trace_to_coordinates
from dijkstra import dijkstra


def points_to_route(
    points: list[tuple[float]],
    image_dimension: tuple[int, int],
    bbox: list[float] = [45.44, -122.63, 45.50, -122.57],
) -> list[tuple[float, float]]:
    """
    Given a list of points, return a route that connects them on map.
    The route is represented by a list of coordinates.
    """
    # construct the graph and node coordinates
    roads = get_roads(bbox)
    graph = construct_graph(roads)
    nodes = construct_node_coordinates(roads)

    image_coords = line_trace_to_coordinates(points, image_dimension, bbox)

    route = []

    for s, e in zip(image_coords[:-1], image_coords[1:]):
        start_node = get_closest_node(s, nodes)
        end_node = get_closest_node(e, nodes)
        path = dijkstra(graph, nodes, start_node, end_node)
        route.extend(path)

    return route
