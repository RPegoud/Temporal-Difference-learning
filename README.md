# Temporal-Difference-learning

This repository contains the code used to generate the experiment described in the article: [Temporal-Difference Learning and the importance of exploration: An illustrated guide](https://medium.com/p/5f9c3371413a), published in **Towards Data Science**.

This project aims to compare several Temporal Difference learning algorithms (Q-learning, Dyna-Q, and Dyna-Q+) in the context of an evolving grid world.

The results obtained during the training outline the importance of continuous exploration in a changing environment and the limits of epsilon-greedy policies as the only source of exploration.

## 🌟 Key Features 
* 🐍 Simple and comprehensive Python (Object Oriented) and Numpy codebase
* 🌐 Dynamic Grid World encouraging continuous exploration
* 🤖 Implemented Algorithms: Q-Learning, Dyna-Q, and Dyna-Q+
* 📊 Plotly graphs enabling state value visualization throughout training and averaged performance reports

## Environment
<div align="center
  <img src='https://miro.medium.com/v2/resize:fit:1400/format:webp/1*1kx0ghZRhWzdEKZUMBKqwQ.jpeg' alt="Environment" width="300" />

<div/>

## Model Performances
  
<!-- markdownlint-disable MD033 -->
<!-- Image row -->
<div align="center">
  <div style="display: flex; justify-content: center;">
    <div style="margin-right: 20px;">
      <img src="https://github.com/RPegoud/Temporal-Difference-learning/blob/main/videos/q_learning_state_values.gif?raw=true" alt="Image 1" width="300" />
      <p align="center"><em>Q-learning</em></p>
    </div>
    <div style="margin-right: 20px;">
      <img src="https://github.com/RPegoud/Temporal-Difference-learning/blob/main/videos/dyna_q_state_values.gif?raw=true" alt="Image 2" width="280" />
      <p align="center"><em>Dyna-Q</em></p>
    </div>
    <div>
      <img src="https://github.com/RPegoud/Temporal-Difference-learning/blob/main/videos/dyna_q_plus_state_values.gif?raw=true" alt="Image 3" width="290" />
      <p align="center"><em>Dyna-Q+</em></p>
    </div>
  </div>
</div>

## Installation

To install and set up the project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/RPegoud/Temporal-Difference-learning.git

2. Navigate to the project directory:

   ```bash
   cd Temporal-Difference-learning

3. Install Poetry (if not already installed):

   ```bash
   python -m pip install poetry

4. Install project dependencies using Poetry:

   ```bash
   poetry install

5. Activate the virtual environment created by Poetry:

   ```bash
   poetry shell

## Use

1. Modify the main.py file depending on the test you want to perform and run:

   ```bash
   python main.py
