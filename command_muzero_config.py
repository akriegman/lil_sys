from easydict import EasyDict

# Reference for config structure:
# LightZero/zoo/box2d/lunarlander/config/lunarlander_disc_muzero_config.py
# startLine: 29
# endLine: 63

config = dict(
    env=dict(
        env_id='docker_command',
        continuous=False,
        manually_discretization=False,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=1e10,
    ),
    policy=dict(
        model=dict(
            block_size=1024,
            vocab_size=50257,
            n_layer=6,
            n_head=8,
            n_embd=512,
            dropout=0.1,
            bias=False
        ),
        cuda=True,
        env_type='terminal',
        game_segment_length=200,
        update_per_collect=50,
        batch_size=64,
        learning_rate=6e-4,
        grad_clip_value=1.0,
        num_simulations=10,
        n_episode=10000,
        eval_freq=1000,
        replay_buffer_size=int(1e6),
    )
)

create_config = dict(
    env=dict(
        type='docker_command',
        import_names=['docker_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='command_muzero'),
) 