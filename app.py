import flask
import predict

app = flask.Flask(__name__, template_folder="templates")

@app.route("/")
def home():
  return flask.render_template('home.html')

@app.route("/graph")
def main():
  return flask.render_template('main.html')

@app.route("/howto")
def howto():
  return flask.render_template('howto.html')

@app.route("/aboutus")
def aboutus():
  return flask.render_template('aboutus.html')

@app.route("/api/v1/predict", methods=['POST'])
def prediction():
  request = flask.request.get_json(force=True)
  if isinstance(request, dict):
    num = request.get("num", 1)
    data = predict.predict()
    response = {}
    for i in range(len(data)):
      response[str(i)] = str(data[i])
  else:
    response = {
      "error": "Invalid JSON"
    }
  return flask.jsonify(response)
