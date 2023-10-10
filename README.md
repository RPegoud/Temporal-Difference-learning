# Temporal-Difference-learning

This repository contains the code used to generate the experiment described in the article: [Temporal-Difference Learning and the importance of exploration: An illustrated guide](https://medium.com/p/5f9c3371413a), published in ***Towards Data Science***.

This project aims to compare several Temporal Difference learning algorithms (Q-learning, Dyna-Q, and Dyna-Q+) in the context of an evolving grid world.

The results obtained during the training outline the importance of continuous exploration in a changing environment and the limits of epsilon-greedy policies as the only source of exploration.

## 🌟 Key Features

* 🐍 Simple and comprehensive Python (Object Oriented) and Numpy codebase
* 🌐 Dynamic Grid World encouraging continuous exploration
* 🤖 Implemented Algorithms: Q-Learning, Dyna-Q, and Dyna-Q+
* 🌍 Model-free and Model-free RL methods comparison
* 📊 Plotly graphs enabling state value visualization throughout training and averaged performance reports
* ✅ Easy installation using Poetry virtual environments

## 🎮 Environment

* The grid is **12 by 8** cells.
* The **agent** starts in the **bottom left** corner of the grid, the objective is to reach the **treasure** located in the **top right** corner (a **terminal state** with reward **1**).
* The **blue portals** are connected, going through the portal located on the cell **(10, 6)** leads to the cell **(11, 0)**. The agent cannot take the portal again after its first transition.
* The **purple portal** only appears **after 100 episodes** but enables the agent to reach the treasure faster. This **encourages continually exploring** the environment.
* The **red portals** are traps (**terminal states** with reward **0**) and end the episode.
* Bumping into a wall causes the agent to remain in the same state.

This experiment aims to compare the behavior of **Q-learning**, **Dyna-Q**, and **Dyna-Q+** agents in a **changing environment**. Indeed, after 100 episodes, the **optimal policy is bound to change** and the optimal number of steps during a successful episode will decrease from **17** to **12**.

<div align="center">
  <img src='https://miro.medium.com/v2/resize:fit:1400/format:webp/1*1kx0ghZRhWzdEKZUMBKqwQ.jpeg' alt="Environment" width="500" />
</div>

## 🏅 Model Performances

*Find the detailed breakdown of model performances and comparisons in the article.*

| Algorithm  | Type        | Updates per step     | Runtime (400 episodes, single CPU) | Discovered optimal strategy (purple portal) | Average cumulative reward |
|:---------- | ----------- | -------------------- |:---------------------------------- | ------------------------------------------- |:------------------------- |
| Q-learning | Model-free  | 1                    | 4.4 sec                            | No                                          | 0.70                      |
| Dyna-Q     | Model-based | 101                  | 31.7 sec                           | No                                          | 0.87                      |
| Dyna-Q+    | Model-based | 101                  | 39.5 sec                           | Yes                                         | 0.79                      |

<div align="center">
  <div style="display: flex; justify-content: center;">
    <div style="margin-right: 20px;">
      <img src="https://miro.medium.com/v2/resize:fit:2000/format:webp/1*zJ-VxFGBRJH-hjoQTeqcDQ.png" alt="Image 1" width="900" />
      <p align="center"><em>Comparison of the cumulative reward per episode averaged over 100 runs</em></p>
    </div>
    <div>
      <img src="https://miro.medium.com/v2/resize:fit:2000/format:webp/1*439rl6tlHwBdp5_BVcbh-Q.png" alt="Image 3" width="900" />
      <p align="center"><em>Comparison of the number of steps per episode averaged over 100 runs</em></p>
    </div>
  </div>
</div>

<!-- markdownlint-disable MD033 -->
<!-- Image row -->
<div align="center">
  <div style="display: flex; justify-content: center;">
    <div style="margin-right: 20px;">
      <img src="https://github.com/RPegoud/Temporal-Difference-learning/blob/main/videos/q_learning_state_values.gif?raw=true" alt="Image 1" width="600" />
      <p align="center"><em>Q-learning</em></p>
    </div>
    <div style="margin-right: 20px;">
      <img src="https://github.com/RPegoud/Temporal-Difference-learning/blob/main/videos/dyna_q_state_values.gif?raw=true" alt="Image 2" width="600" />
      <p align="center"><em>Dyna-Q</em></p>
    </div>
    <div>
      <img src="https://github.com/RPegoud/Temporal-Difference-learning/blob/main/videos/dyna_q_plus_state_values.gif?raw=true" alt="Image 3" width="600" />
      <p align="center"><em>Dyna-Q+</em></p>
    </div>
  </div>
</div>

## 💾 Installation

To install and set up the project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/RPegoud/Temporal-Difference-learning.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Temporal-Difference-learning
   ```

3. Install Poetry (if not already installed):

   ```bash
   python -m pip install poetry
   ```

4. Install project dependencies using Poetry:

   ```bash
   poetry install
   ```

5. Activate the virtual environment created by Poetry:

   ```bash
   poetry shell
   ```

## Use

1. Modify the main.py file depending on the test you want to perform and run:

   ```bash
   python main.py

## 📖 References

> Sutton, R. S., & Barto, A. G. . [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2018), Cambridge (Mass.): The MIT Press.
