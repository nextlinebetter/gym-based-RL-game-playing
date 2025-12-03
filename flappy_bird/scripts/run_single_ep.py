import gymnasium as gym
import flappy_bird  # for registry


env_name = "FlappyBirdEnvWithContinuousObs"

# Create our training environment
env = gym.make(env_name, render_mode="human")

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see"
# info: extra debugging information (usually not needed for basic learning)

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    print(observation)
    # Choose an action randomly
    action = env.action_space.sample()

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # terminated: True if agent failed
    # truncated: True if we hit the time limit

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
