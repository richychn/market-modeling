# pylint: disable=C0111

import flask
import predict

APP = flask.Flask(__name__, template_folder="templates")

@APP.route("/")
def home():
    return flask.render_template('main.html')

@APP.route("/howto")
def howto():
    return flask.render_template('howto.html')

@APP.route("/aboutus")
def aboutus():
    return flask.render_template('aboutus.html')

@APP.route("/api/v1/predict_levels", methods=['POST'])
def prediction_levels():
    request = flask.request.get_json(force=True)
    if isinstance(request, dict):
        num = request.get("num", 1)
        if num == "":
            num = 0
        data = predict.predict_levels(int(num))
        response = {}
        for i,_ in enumerate(data):
            response[str(i)] = str(data[i])
    else:
        response = {
            "error": "Invalid JSON"
        }
    return flask.jsonify(response)

@APP.route("/api/v1/predict_growth", methods=['POST'])
def prediction_growth():
    request = flask.request.get_json(force=True)
    if isinstance(request, dict):
        num = request.get("num", 1)
        if num == "":
            num = 0
        data = predict.predict_growth(int(num))
        response = {}
        for i,_ in enumerate(data):
            response[str(i)] = str(data[i])
    else:
        response = {
            "error": "Invalid JSON"
        }
    return flask.jsonify(response)
