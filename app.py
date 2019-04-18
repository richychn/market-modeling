import flask
import predict

app = flask.Flask(__name__, template_folder="templates")

@app.route("/")
def home():
  return flask.render_template('main.html')

@app.route("/howto")
def howto():
  return flask.render_template('howto.html')

@app.route("/aboutus")
def aboutus():
  return flask.render_template('aboutus.html')

@app.route("/api/v1/predict_levels", methods=['POST'])
def prediction_levels():
  request = flask.request.get_json(force=True)
  if isinstance(request, dict):
    num = request.get("num", 1)
    data = predict.predict_levels(int(num))
    response = {}
    for i in range(len(data)):
      response[str(i)] = str(data[i])
  else:
    response = {
      "error": "Invalid JSON"
    }
  return flask.jsonify(response)

@app.route("/api/v1/predict_growth", methods=['POST'])
def prediction_growth():
  request = flask.request.get_json(force=True)
  if isinstance(request, dict):
    num = request.get("num", 1)
    data = predict.predict_growth(int(num))
    response = {}
    for i in range(len(data)):
      response[str(i)] = str(data[i])
  else:
    response = {
      "error": "Invalid JSON"
    }
  return flask.jsonify(response)

