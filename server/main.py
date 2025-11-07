from flask import Flask, send_from_directory
from analyze import all_in_one

app = Flask(__name__)
@app.route("/analyze/<stock_code>")
def analyze(stock_code):
    return all_in_one(stock_code)


@app.route("/<path:filename>")
def index(filename):
    return send_from_directory('app', filename)

if __name__ == '__main__':
    app.run()