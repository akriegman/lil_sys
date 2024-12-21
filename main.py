import sys
from pathlib import Path

# Add submodule paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "LightZero"))
sys.path.append(str(project_root / "nanoGPT"))

from docker_env import DockerCommandEnv
from lzero.entry import train_muzero
from command_muzero_config import config, create_config

def main():
    train_muzero(config, create_config)

if __name__ == "__main__":
    main() 
