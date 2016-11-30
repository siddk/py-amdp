# PY-AMDP
Repository containing all Python machine translation code for AMDP Language Grounding Research.

### Directory Structure ###
The following details each of the repository directory, and describes the relevant format for adding
data.

  + data/ - Contains all training/test data for running the translation models. 
    - machine-commands.txt - Mapping file containing all possible machine language commands, at
                             each of the three levels of abstractions.
    - parallel/ - Directory containing weakly aligned parallel data. File pairs should be labelled
                  <id>.en and <id>.ml, where a '.en' file extension denotes the natural language
                  half of the parallel data, and a '.ml' file extension denotes the machine language
                  half of the parallel data.
  
  + models/ - Directory containing Python implementations of the various machine translation models.
    - ibm2.py - IBM Model 2 Machine Translation Implementation.
    
  + preprocessor/ - Directory containing preprocessors for the necessary models - loads and parses
                    the parallel data into the relevant structures necessary for each respective 
                    model.
                    
### Data Log ###
The following records how much data has been collected, as well as the results. Should be added to
as more data is collected:


              
