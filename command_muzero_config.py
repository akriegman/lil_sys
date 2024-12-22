from easydict import EasyDict

# Main configuration
main_config = dict(
    exp_name='command_muzero',
    env=dict(
        n_action=4,  # Adjust based on your command environment
        save_replay_episodes=1,
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 64, 64),  # Adjust based on your observation space
            action_space_size=4,  # Should match n_action
            lstm_hidden_size=512,
            representation_network_hidden_size=256,
            dynamics_network_hidden_size=256,
            prediction_network_hidden_size=256,
        ),
    ),
)
main_config = EasyDict(main_config)

# Create configuration
create_config = dict(
    env=dict(
        type='command',  # Your environment type
        import_names=['docker_env'],  # Path to your environment
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='muzero',  # Using standard MuZero algorithm
        import_names=['lzero.policy.muzero'],
    ),
)
create_config = EasyDict(create_config) 