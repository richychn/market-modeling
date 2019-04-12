import flask
import predict
from keras import backend

app = flask.Flask(__name__, template_folder="templates")

@app.route("/")
def home():
  return flask.render_template('home.html')

@app.route("/graph")
def main():
  data = predict.predict()
  backend.clear_session()
  return flask.render_template('main.html', data=data)

@app.route("/howto")
def howto():
  return flask.render_template('howto.html')

@app.route("/aboutus")
def aboutus():
  return flask.render_template('aboutus.html')

@app.route("/api/v1/predict", methods=['POST'])
def prediction():
  request = flask.request.get_json(silent=True)
  if isinstance(request, dict):
      num = request.get("num", 1)
      data = predict.predict(num)
      response = {
        "data": data
      }
  else:
    response = {
      "error": "Invalid JSON"
    }
  return flask.jsonify(response)
