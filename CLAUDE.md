## Guidelines for this project.
This project is a modified version of Verl, a reinforcement learning framework.
Specifically, I want to implement the run-ahead rollout strategy in Verl's rollout. 

The idea behind is that common rollout will have a lot of GPU bubble time, steps from the waiting for a subset of samples in a batch that have not finished yet. 


## Goal
Have a proper implementation of run-ahead rollout strategy in Verl's rollout. 

## Reference 
See `./claude_docs` for some background information of agentLoop and how the vLLM server rollout works.


## Guide for executing the implemented function in our environment
+ All the dependency has been installed. Get the virtual environments by `source .venv/bin/activate`