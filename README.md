# MCTS - Connect four

This is my project for the Markov Decision Processes and Reinforcement Learning course of Trento University in 2024/2025.

## Description

This project implements and trains three different reinforcement learning algorithms on the game of Connect Four: Naive Monte Carlo Tree Search (MCTS), REINFORCE with baseline, and AlphaZero. The objective is to explore the MCTS technique, compare it with policy-based methods, and apply it to a real game.


## Project Development Process

1. Project selection
For this project, my initial goal was to focus on an implementation-based approach. Among the proposed topics, the one that interested me the most was "Monte Carlo Tree Search and AlphaGo." After reviewing the corresponding DRIVE papers, I became particularly inclined to explore the MCTS technique further, given its effectiveness in decision-making and strategic planning.

2. Framework selection and game selection
From some time I was interested to try the JAX machine learning framework, while I was checking out some library for reinforcement learning and I found first mctx, an efficient implementation of Monte Carlo Tree Search and the pgx which provides various environment. I then read the paper of pgx and select the game Connect Four based on its manageable complexity.

3. Algorithm Implementation
The first algorithm I implemented was a basic MCTS variant that used a uniform prior and approximated state values through a single random rollout. This simple approach was valuable for gaining a deeper understanding of how JAX, pgx, and mctx function in practice.
Next, I implemented AlphaZero, using a ResNet for policy and value approximation, and successfully trained the model. Given the differences in performance and complexity between these methods, I decided to also implement REINFORCE with baseline to evaluate a policy-based approach without MCTS. However, despite improvements to the loss function, I was unable to successfully train the model.

4. More training experiment and manual evaluation
I focused on improving the performance of the AlphaZero algorithm by experimenting with various parameters for a better training, such as batch size, the number of MCTS simulations, and the neural network architecture, including the number of channels and layers. To assess the effectiveness of these changes, I manually tested some games and also test self-play between different models to evaluate their performance.

5. Notebook and presentation
During the development of all the algorithms, I used Jupyter notebooks to facilitate the implementation and experimentation process. I then placed the most significant aspects of each algorithm into a single notebook. And then I wrote the  presentation.





