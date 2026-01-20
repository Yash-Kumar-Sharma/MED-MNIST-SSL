import pytorch_lightning as pl
import torch
import hydra
from omegaconf import OmegaConf,DictConfig
import numpy as np
import random

from Pretraining.Pretraining import Pretraining
from Linear_Evaluation.Transfer_Learning import Transfer_Learning
from Linear_Evaluation.Redefine import Redefine
from Linear_Evaluation.Linear_Evaluation import Linear_Evaluation

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    #device = torch.device('cuda')
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    pl.seed_everything(42, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache() 
    if(cfg.transfer_learning.transfer_learning):
        Transfer_Learning(cfg)
    else:
        trained_model = Pretraining(cfg) 
        if(cfg.model.name != "Supervised"):
            Linear_Evaluation(trained_model, cfg)

if __name__ == "__main__":
    main()
