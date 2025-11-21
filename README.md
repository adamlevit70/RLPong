# RLPong
This project trains an AI agent to play Pong using Deep Q-Learning (DQN).
The agent watches the game state (ball position, paddle position, and their velocity) and learns how to react by trial and error. Over time, it improves by maximizing rewards for hitting the ball and scoring points, while getting punished for missing.

## The implementation includes:

A Pong environment built with PyGame.

A neural network policy written in PyTorch.

Experience Replay memory.

A Target Network for stable learning.

Parallel simulation of multiple games for faster training.

A live gameplay viewer to watch the agent fail, learn, and eventually play well in different episodes.

The goal is to train a single agent to compete against a built-in opponent and eventually learn effective strategies.
