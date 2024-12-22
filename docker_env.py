from typing import Dict, Any
import docker
import tempfile
import torch
from nanoGPT.model import GPT, GPTConfig
from ding.envs import BaseEnv
from ding.envs.env.base_env import ENV_REGISTRY
import numpy as np

class DockerCommandEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.client = docker.from_env()
        self.container = None
        self.terminal_buffer = ""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize GPT model for policy/reward
        self.chars = list(set(' '.join([chr(i) for i in range(32, 127)])))  # ASCII printable chars
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Update GPTConfig with new vocab size
        self.config = GPTConfig(
            block_size=1024,
            vocab_size=self.vocab_size,
            n_layer=6,
            n_head=8,
            n_embd=512,
            dropout=0.1,
            bias=False
        )
        self.model = GPT(self.config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=6e-4,
            weight_decay=1e-1
        )
        
    def reset(self):
        if self.container:
            self.container.stop()
            self.container.remove()
            
        self.container = self.client.containers.run(
            "alpine:latest",
            detach=True,
            tty=True
        )
        self.terminal_buffer = ""
        return self._get_observation()
        
    def step(self, command: str):
        try:
            # Execute command and get output
            exit_code, output = self.container.exec_run(command)
            
            # Update terminal buffer
            self.terminal_buffer += f"$ {command}\n{output.decode()}\n"
            
            # Calculate reward based on language model improvement
            reward = self._calculate_reward()
            
            done = len(self.terminal_buffer) >= 10000  # Max buffer size
            return self._get_observation(), reward, done, {}
            
        except Exception as e:
            return self._get_observation(), -1, True, {"error": str(e)}
            
    def _get_observation(self):
        return {
            "terminal": self.terminal_buffer
        }
        
    def _calculate_reward(self):
        # Encode the terminal buffer
        tokens = torch.tensor([[self.char_to_idx.get(c, 0) for c in self.terminal_buffer]])
        
        # Calculate initial loss
        self.model.train()
        with torch.no_grad():
            initial_logits = self.model(tokens)
            initial_loss = torch.nn.functional.cross_entropy(
                initial_logits.view(-1, initial_logits.size(-1)), 
                tokens.view(-1)
            )
        
        # Update model
        logits = self.model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tokens.view(-1)
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Reward is the improvement in loss
        reward = float(initial_loss - loss)
        return reward
        
    def close(self):
        # Clean up any resources
        pass

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(seed)

    def __repr__(self) -> str:
        return "DockerCommandEnv"

    @property
    def observation_space(self):
        # Define your observation space
        pass

    @property
    def action_space(self):
        # Define your action space
        pass

    @property
    def reward_space(self):
        # Define reward space if needed
        pass

# Register the environment
ENV_REGISTRY.register('docker_command', DockerCommandEnv)
        
