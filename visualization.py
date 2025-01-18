import folium
import gpxpy
import re
import os
from dijkstra import dijkstra
from OSM import get_roads, construct_graph, construct_node_coordinates, get_closest_node
from shape_processor import image_to_coordinates

bbox = [45.44, -122.63, 45.50, -122.57]
roads = get_roads(bbox)
graph = construct_graph(roads)
nodes = construct_node_coordinates(roads)

image_filename = "images/pi.jpg"
image_name = re.search(r"([^\/]+)(?=\.[^.]+$)", image_filename).group(1)
image_coords = image_to_coordinates(image_filename, bbox)


def create_gpx_file(coords, filename):
    gpx = gpxpy.gpx.GPX()

    # Create first track in our GPX:
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    # Create first segment in our GPX track:
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    for coord in coords:
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(coord[0], coord[1]))

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(filename, "w") as f:
        f.write(gpx.to_xml())

full_route = []
# Create a map centered at the given latitude and longitude
m = folium.Map(location=[45.467, -122.637], zoom_start=15)

for s, e in zip(image_coords[:-1], image_coords[1:]):
    start_node = get_closest_node(s, nodes)
    end_node = get_closest_node(e, nodes)
    path = dijkstra(graph, nodes, start_node, end_node)
    folium.CircleMarker(
        location=nodes[start_node],
        radius=10,
        color="red",
        fill=True,
    ).add_to(m)
    folium.CircleMarker(
        location=nodes[end_node],
        radius=10,
        color="red",
        fill=True,
    ).add_to(m)
    folium.PolyLine(path).add_to(m)
    full_route.extend(path)

gpx_filename = f"routes/{image_name}.gpx"
create_gpx_file(full_route, gpx_filename)

# start_node = 1710336791
# end_node = 454072437
# path = dijkstra(graph, coordinates, start_node, end_node)
# print('Path:', path)

# start_coords = coordinates[start_node]
# end_coords = coordinates[end_node]

# folium.CircleMarker(
#     location=start_coords,
#     radius=10,
#     color="red",
#     fill=True,
#     popup="Start",
# ).add_to(m)
# folium.CircleMarker(
#     location=end_coords,
#     radius=10,
#     color="blue",
#     fill=True,
#     popup="End",
# ).add_to(m)

# folium.PolyLine(path).add_to(m)
# folium.PolyLine([start_coords, end_coords], color="red").add_to(m)

# folium.PolyLine(image_coords).add_to(m)

m.save(f"routes/{image_name}.html")
