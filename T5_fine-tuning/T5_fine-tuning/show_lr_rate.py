import configs
import utils

from transformers import T5ForConditionalGeneration
import torch.optim as optim


model_name = 't5-small'

config = configs.Config()

model = T5ForConditionalGeneration.from_pretrained(model_name)

optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), 
                            eps=1e-6, weight_decay=config.weight_decay)
lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.t0)

utils.plot_learning_rate(optimizer, lr_scheduler, config.epoch)