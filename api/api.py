"""
api.py 

Starts a lightweight Flask API Server, that can be queried via calls to CURL
"""
from flask import Flask, request
from flask_restful import Resource, Api
import run_ibm
import run_nn
import run_rnn
import sys
import tensorflow as tf

app = Flask(__name__)
api = Api(app)

### CHANGE THIS: [ibm, nn, rnn] ###
model_type = 'ibm' 

# Model Switch Code
if model_type == 'rnn':
    m = run_rnn.load_model()
elif model_type == 'nn':
    m = run_nn.load_model()
elif model_type == 'ibm':
    m, j = run_ibm.load_model()
else:
    raise UserWarning("Model not defined!")

class ModelInterface(Resource):
    def put(self):
        nl_command = request.form['command']
        if model_type == 'rnn':
            rf, _, lvl, _ = m.score(nl_command.split())
        elif model_type == 'nn':
            rf, _, lvl, _ = m.score(nl_command.split())
        elif model_type == 'ibm':
            lvl, rf = run_ibm.score(m, j, nl_command.split())
        return {"Level": lvl, "Reward Function": " ".join(rf)}

api.add_resource(ModelInterface, '/model')

if __name__ == "__main__":
    app.run()