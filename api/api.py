"""
api.py 

Starts a lightweight Flask API Server, that can be queried via calls to CURL
"""
from flask import Flask, request, jsonify
import run_ibm
import run_nn
import run_single_rnn
import sys
import tensorflow as tf

app = Flask(__name__)

### CHANGE THIS: [ibm, nn, rnn] ###
model_type = 'rnn' 

# Model Switch Code
if model_type == 'rnn':
    m = run_single_rnn.load_model()
elif model_type == 'nn':
    m = run_nn.load_model()
elif model_type == 'ibm':
    m, j = run_ibm.load_model()
else:
    raise UserWarning("Model not defined!")

@app.route('/model')
def model():
    nl_command = request.args.get('command')
    if model_type == 'rnn':
        print nl_command
        rf, _ = m.score(nl_command.lower().split())
        lvl = rf[0]
        rf = rf[1:]
    elif model_type == 'nn':
        rf, _, lvl, _ = m.score(nl_command.lower().split())
    elif model_type == 'ibm':
        lvl, rf = run_ibm.score(m, j, nl_command.lower().split())
    return str(lvl) + " " + " ".join(rf) + "\n"

if __name__ == "__main__":
    app.run()