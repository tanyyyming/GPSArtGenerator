from flask import Flask, jsonify, request
from flask_cors import CORS
import api

app = Flask(__name__)
CORS(app)


@app.route("/")
def echo():
    return "Hi!", 200


@app.route("/api/points_to_route", methods=["POST"])
def points_to_route():
    try:
        data = request.json

        if not all(key in data for key in ["lines", "width", "height"]):
            return jsonify({"error": "Invalid request"}), 400

        dimension = (data["height"], data["width"])
        bbox = data["bbox"] if "bbox" in data else [45.44, -122.63, 45.50, -122.57]

        routes = []

        for line in data["lines"]:
            # points are in the form [{x: 1, y: 2}, {x: 3, y: 4}]
            # convert the points to a list of tuples
            points = [(point["x"], point["y"]) for point in line["points"]]
            route = api.points_to_route(points, dimension, bbox)
            routes.append(route)

        return jsonify({"routes": routes}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8888)
