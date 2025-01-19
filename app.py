from flask import Flask, render_template, request, jsonify
import time
import logging
import datetime

# app will run at: http://127.0.0.1:5000/

# Load environment variables from a .env file


# Set up logging in the app.log file
log = logging.getLogger("assistant")
logging.basicConfig(filename="app.log", level=logging.INFO)

# Import and configure OpenAI
from openai import OpenAI

client = OpenAI()

# Flask app setup
app = Flask(__name__)

# Render the HTML template - we're going to see a UI!!!
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
  
# Function to add to the log in the app.log file
def log_run(run_status):
    if run_status in ["cancelled", "failed", "expired"]:
        log.error(str(datetime.datetime.now()) + " Run " + run_status + "\n")

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
