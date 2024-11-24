# DQN Agent for Atari Breakout

This project implements a Deep Q-Network (DQN) agent to play Atari's Breakout game using Stable-Baselines3 and Gymnasium. The implementation includes both training and evaluation scripts.

## Project Structure

```
DQN-With-Atari/
│
├── train.py          # Script for training the DQN agent
├── play.py           # Script for evaluating the trained agent
├── README.md         # This file
│
├── models/           # Directory for saved models (created during training)
│   └── policy.zip    # Trained model file (created after training)
│
└── logs/            # Training logs for tensorboard (created during training)
```

## Prerequisites

### System Requirements
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Required Python Packages
- gymnasium[atari]
- stable-baselines3[extra]
- ale-py

## Installation

1. Clone this repository:
```bash
git clone https://github.com/DavidkingMazimpaka/DQN-With-Atari.git
cd DQN-With-Atari
```

2. Create and activate a virtual environment (recommended):
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install gymnasium[atari] stable-baselines3[extra] ale-py
```

4. Install Atari ROMs:
```bash
pip install autorom[accept-rom-license]
```

## Usage

### Training the Agent

To train the DQN agent, run:
```bash
python train.py
```

The training script will:
- Create a 'models/' directory to save checkpoints
- Create a 'logs/' directory for tensorboard logs
- Train the agent for 50,000 timesteps (configurable in the script)
- Save the final model as 'models/policy.zip'
- Save periodic checkpoints during training

Training progress will be displayed in the console with metrics such as:
- Episode reward mean
- Episode length mean
- Learning rate
- Exploration rate

### Playing with the Trained Agent

To watch the trained agent play, run:
```bash
python play.py
```

This will:
- Load the trained model from 'models/policy.zip'
- Run 5 episodes of the game with visualization
- Display the total reward and steps for each episode

## Monitoring Training Progress

You can monitor the training progress using Tensorboard:

1. Install tensorboard if not already installed:
```bash
pip install tensorboard
```

2. Run tensorboard:
```bash
tensorboard --logdir logs/
```

3. Open your web browser and go to `http://localhost:6006`

## Configuration

### Training Parameters
You can modify the following parameters in `train.py`:
- `total_timesteps`: Number of training steps (default: 50,000)
- `learning_rate`: Learning rate for the neural network (default: 1e-4)
- `buffer_size`: Size of the replay buffer (default: 50,000)
- `learning_starts`: Number of steps before starting training (default: 1,000)
- `batch_size`: Size of training batches (default: 32)
- `exploration_fraction`: Fraction of total timesteps for exploration (default: 0.1)

### Evaluation Parameters
In `play.py`, you can modify:
- `n_episodes`: Number of episodes to play (default: 5)

## Troubleshooting

1. If you encounter ROM-related errors:
```bash
pip install autorom[accept-rom-license]
```

2. If you get rendering errors:
- Make sure you have the required system dependencies for OpenGL
- Try updating your graphics drivers

3. If the model file isn't found:
- Ensure you've run `train.py` before `play.py`
- Check that the model file exists in the 'models/' directory

## Performance Notes

- The default training duration (50,000 steps) is relatively short for optimal performance
- For better results, consider increasing `total_timesteps` to 500,000 or more
- Training time will vary depending on your hardware
- GPU acceleration is supported and recommended for faster training

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for the DQN implementation
- [Gymnasium](https://gymnasium.farama.org/) for the Breakout environment
- Atari and Breakout are trademarks of Atari Interactive Inc.

## Author
- [KingDavid Mazimpaka](https://github.com/DavidkingMazimpaka/)
