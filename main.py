import sys
from pathlib import Path

# Add submodule paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "LightZero"))
sys.path.append(str(project_root / "nanoGPT"))

from docker_env import DockerCommandEnv
from lm_wrapper import LanguageModelWrapper
from command_policy import CommandMuZero

def main():
    env = DockerCommandEnv()
    lm = LanguageModelWrapper()
    
    # Configure MuZero
    muzero_config = {
        "env": env,
        "policy": CommandMuZero,
        "policy_kwargs": {"lm_wrapper": lm},
        # Add other MuZero configs based on LightZero framework
    }
    
    # Train agent
    agent = CommandMuZero(muzero_config)
    agent.train()

if __name__ == "__main__":
    main() 