import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
import tensorflow as tf
import ale_py

gym.register_envs(ale_py)


def create_env(render_mode=None):
    """Create and wrap the Breakout environment"""
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
    env = Monitor(env)
    env = AtariWrapper(env, screen_size=84)  # Adjusted to remove grayscale_obs
    return env


def main():
    # saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create and wrap the env
    env = DummyVecEnv([lambda: create_env()])

    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: create_env()])

    # Initialize the DQN agent with CnnPolicy
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=10000,  # Size of the replay buffer
        learning_starts=1000,  # steps before starting training
        batch_size=32,
        gamma=0.99,  # Discount factor
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        train_freq=4,  # Update the model every 4 steps
        gradient_steps=1,
        target_update_interval=1000,  # How often to update target network
        verbose=1,
        tensorboard_log="logs/"
    )

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/",
        name_prefix="dqn_breakout"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_model",
        log_path="logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Train the agent
    total_timesteps = 50000  # no of steps just as asked in the work
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    # Save the final model
    model.save("models/policy.zip")  # .zip for clarity

    print("Training completed! Model saved as 'policy.zip'")

if __name__ == "__main__":
    main()





