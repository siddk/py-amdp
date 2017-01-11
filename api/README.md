# API Documentation #

There are a few ways to run the API, and all the relevant code is contained in
this sub-directory.

First, note that in each of the *_ckpt checkpoint directories, are serialized forms of each
of the three different models: the IBM Model 2 (ibm\_ckpt), the Feed-Forward Neural
Network (nn\_ckpt), and the Recurrent Neural Network (rnn\_ckpt). 

To train up a new model, you can use the run\_'model'.py files. First, though, move (or delete) the 
existing subdirectories, and create a new empty directory called 'model'\_ckpt/. Then, run the 
respective run\_'model'.py file (for example, 'python run\_rnn.py').

There are two ways to run existing pre-trained models. The first is via a call to run_'model'.py. 
This will load the model, and drop you into a REPL, where you can enter natural language commands
(space-separated, for example: "Go to the red room ."), and get back both the predicted AMDP level,
as well as the predicted Reward Function (for example, "agentInRegion agent0 room0").

Additionally, there is a lightweight Flask server that can be run, that loads the model into an API
that can be queried via calls to CURL. To run this, first go into the file "api.py", and change the
line marked "CHANGE ME" to the desired model-type (one of 'rnn', 'nn', or 'ibm'). Then, start the 
Flask application via a call to 'python api.py'. This should load a Flask application on your localhost
port 5000.

You can then query the API via a call to CURL as follows:

curl http://localhost:5000/model -d "command=Go to the red room" -X PUT

Note that you enter a natural language command via -d "command=COMMAND GOES HERE".