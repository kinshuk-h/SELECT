from .apo import AbstentionWithDPOTraining
from .aft import AbstentionWithSFTTraining

def get_approach(name):
    algorithm = name.split('-')[-1]
    if algorithm == "sft-dpo": return AbstentionWithDPOTraining(post_sft_checkpoint=True)
    elif algorithm == "dpo"  : return AbstentionWithDPOTraining()
    elif algorithm == "sft"  : return AbstentionWithSFTTraining()
    raise ModuleNotFoundError(name)

__all__ = [ "AbstentionWithDPOTraining", "get_approach" ]