import gymnasium as gym
from stable_baselines3 import DQN
import ale_py
import numpy as np
from gym import spaces
from PIL import Image


class CustomAtariWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(CustomAtariWrapper, self).__init__(env)
        # Define the observation space to match the expected shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

    def observation(self, obs):
        img = Image.fromarray(obs)
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((84, 84), Image.BILINEAR)
        return np.array(img).astype(np.uint8).reshape(1, 84, 84)


def main():
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = CustomAtariWrapper(env)  # Wrap the environment

    # Load the trained model
    try:
        model = DQN.load("models/policy.zip")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have run train.py first and the model file exists.")
        return

    # Run several episodes
    n_episodes = 5
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            # Predicting the best action (greedy policy)
            action, _ = model.predict(obs, deterministic=True)

            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"Episode {episode + 1} - Total Reward: {total_reward} - Steps: {steps}")

    env.close()


if __name__ == "__main__":
    main()