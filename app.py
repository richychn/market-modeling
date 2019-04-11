import flask

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

@app.route("/api/v1/greeting", methods=['POST'])
def greeting_api():
  request = flask.request.get_json(silent=True)
  if isinstance(request, dict):
    response = {
      "greeting": "Hello, " + request.get("name", "friend") + "!"
    }
  else:
    response = {
      "error": "Invalid JSON"
    }
  return flask.jsonify(response)
