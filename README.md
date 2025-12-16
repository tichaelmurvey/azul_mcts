# AI Azul Player and Game Interface
🟥🟪🟦🟩🟨🟧

This repo contains code for an AI player using the Monte Carlo Tree Search Method.
It also contains a human-readable text interface for Azul, allowing users to play against the AI.

## How to Use

### Web version
You can play against a [standard version of the AI online](https://tichaelmurvey.github.io/azul_mcts/).

### Use locally and edit
To use the AI locally, try different model parameters, or improve the system, you can clone this repo.

**AzulMCTest.py** contains the human/AI gameplay method. Run this file to play against the default AI.

**AzulMCTestPlayers.py** contains definitions for simple opponents used to test the AI model. RandomPlayer chooses random moves. HeuristicPlayer chooses moves based on a very simple "Which move gives me the most immediate point potential" policy.

## Model Adjustment
**AzulMCHeuristic.py** contains the Monte Carlo Tree Search agent.
The following params can be used to modify its behaviour:
**Iterations**: How many moves will the model explore to check outcomes? Can be overridden with a time limit per move.
**Time limit**: Overrides Iterations. Instead of exploring a number of moves, the model will explore until the time runs out.
**Terminus evaluator**: How should the model judge the value of end states? Uses heuristic_score by default, but other options can be defined.
**Opponent simulated strategy**: How should the model predict opponent moves when simulating outcomes?
**Filter floor**: Should the model consider moves that move directly to the floor? (These moves are almost never useful)
