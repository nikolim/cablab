def track_reward(reward, saved_rewards):
    saved_rewards = list(saved_rewards)
    if reward == -1:
        saved_rewards[0] += 1
    if reward == -10:
        saved_rewards[1] += 1
    if reward == -5:
        saved_rewards[2] += 1
    return tuple(saved_rewards)

def log_rewards(writer, saved_rewards, episode_reward, episode):
    writer.add_scalar("Path Penalty", saved_rewards[0], episode)
    writer.add_scalar("Illegal Pick-up / Drop-off", saved_rewards[1], episode)
    writer.add_scalar("Illegal Move", saved_rewards[2], episode)
    writer.add_scalar("Reward", episode_reward, episode)


def log_reward_epsilon(writer, episode_reward, epsilon, episode):
    writer.add_scalars(
        "Reward/Epsilon", {"Reward": episode_reward, "Epsilon": epsilon * 1000}, episode
    )


def log_reward_uncertainty(writer, episode_reward, uncertainty, episode):
    writer.add_scalars(
        "Reward/Uncertainty",
        {"Reward": episode_reward, "Uncertainty": uncertainty},
        episode,
    )
