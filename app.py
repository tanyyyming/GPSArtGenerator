from flask import Flask, jsonify, request
import api

app = Flask(__name__)


@app.route("/image-to-route", methods=["POST"])
def image_to_route():
    try:
        data = request.json

        if not all(key in data for key in ["lines", "width", "height"]):
            return jsonify({"error": "Invalid request"}), 400

        dimension = (data["height"], data["width"])

        routes = []

        for line in data["lines"]:
            # points are in the form [{x: 1, y: 2}, {x: 3, y: 4}]
            # convert the points to a list of tuples
            points = [(point["x"], point["y"]) for point in line["points"]]
            route = api.points_to_route(points, dimension)
            routes.append(route)

        return jsonify({"routes": routes}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
