from lzero.policy import MuZeroPolicy
import torch

@POLICY_REGISTRY.register('command_muzero')
class CommandMuZeroPolicy(MuZeroPolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.chars = list(set(' '.join([chr(i) for i in range(32, 127)])))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
    def _forward_collect(self, data):
        obs = data['obs']
        terminal = obs['terminal']
        
        # Generate next command using character-level tokens
        tokens = torch.tensor([[self.char_to_idx.get(c, 0) for c in terminal + "$"]])
        
        with torch.no_grad():
            output = self.model.generate(
                tokens,
                max_new_tokens=50,
                temperature=0.7,
                top_k=50,
                eos_token_id=self.char_to_idx.get('\n', 0)
            )
        
        command = ''.join([self.idx_to_char[i.item()] for i in output[0]])
        command = command.split("$")[-1].strip()
        
        return command 