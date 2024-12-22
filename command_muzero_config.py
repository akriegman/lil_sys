from easydict import EasyDict

# Main configuration
main_config = dict(
    exp_name='command_muzero',
    env=dict(
        n_action=4,  # Adjust based on your command environment
        save_replay_episodes=1,
    ),
    policy=dict(
        # Device configurations
        cuda=True,
        model_path=None,
        # Model configurations
        model=dict(
            observation_shape=(3, 64, 64),  # Adjust based on your observation space
            action_space_size=4,  # Should match n_action
            lstm_hidden_size=512,
            representation_network_hidden_size=256,
            dynamics_network_hidden_size=256,
            prediction_network_hidden_size=256,
        ),
        # Training configurations
        learn=dict(
            update_per_collect=50,
            batch_size=256,
            learning_rate=0.003,
            target_update_freq=100,
        ),
        # Collection configurations
        collect=dict(
            n_episode=8,
            env_num=8,
            n_step=200,
        ),
    ),
)
main_config = EasyDict(main_config)

# Create configuration
create_config = dict(
    env=dict(
        type='docker_command',
        import_names=['docker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
create_config = EasyDict(create_config) 