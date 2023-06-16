# Temporal-Difference-learning

This project aims to compare several Temporal Difference learning algorithms (Q-learning, Dyna-Q and Dyna-Q+) in the context of an evolving grid world.

The results obtained during the training outline the importance of continous exploration in a changing environment and the limits of epsilon-greedy policies as only source of exploration.

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
