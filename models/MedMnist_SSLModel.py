from torch import nn, optim
import pytorch_lightning as pl
import torch
import torchvision
from Loss.loss import xent_loss
import torchmetrics
import os
import utils
from Pretraining.Knn_Monitor import Knn_Monitor
from copy import deepcopy
import math
    
class MedMnist_SSLModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()
        #self.automatic_optimization = False
        self.net = utils.GetBackbone(config.backbone.name, config.dataset.name, one_color_channel=config.dataset.one_color_channel)
        self.hidden_size = config.backbone.feature_size
        self.projection_size = config.backbone.feature_size // 4

        self.head = nn.Sequential(
            nn.Linear(config.backbone.feature_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.projection_size),
            nn.BatchNorm1d(self.projection_size),
        )
        
        self.encoder_x = nn.Sequential(self.net,self.head)
        self.encoder_c = deepcopy(self.encoder_x)
        
        self._init_encoder_c()
        self.K = config.model.dict_K
        self.image_size = config.dataset.image_size
        self.save_path = config.dataset.save_path
        self.dataset = config.dataset.name
        self.filepath = config.dataset.filepath
        self.batch_size = config.training.batch_size
        
        self.checkpoint_dir = utils.GetCheckpointDir(config, train_mode="pretrain")
        self.lr = config.training.lr
        self.temperature = config.model.temperature
        self.weight_decay = config.training.weight_decay
        self.epochs = config.training.max_epochs
        self.warmup_epochs = config.training.warmup_epochs
        self.warmup_lr = config.training.warmup_lr

        self.model_name = config.model.name
        self.m = config.model.m
        self.backbone = config.backbone.name
        self.channel = 3
        if(config.dataset.one_color_channel):
            self.channel = 1

        self.accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes = config.dataset.num_classes)
        self.loss_metric = torchmetrics.MeanMetric()
        self.outputs = []
        self.alignment = 0.0
        self.uniformity = 0.0
        self.total = 0
        self.knn_monitor = Knn_Monitor(config)  
        self.loss_value = []
        

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x

     
    @torch.no_grad()
    def _init_encoder_c(self):
        for param_normalize, param_transform in zip(self.encoder_x.parameters(), self.encoder_c.parameters()):
            param_transform.data.copy_(param_normalize.data)
            param_transform.requires_grad = False

    @torch.no_grad()
    def _momentum_update_encoder_c(self):
        for param_normalize, param_transform in zip(self.encoder_x.parameters(), self.encoder_c.parameters()):
            param_transform.data = param_transform.data * self.m + param_normalize.data * (1.0 - self.m)

    def training_step(self, batch, batch_idx): 

        train_x, train_x_transform = batch
        
        set_n = train_x[::2];                    '''x'''
        set_n_counter = train_x[1::2];           '''y'''
        set_a_counter = train_x_transform[::2];  '''y\prime'''
        set_a = train_x_transform[1::2];         '''x\prime'''

        self.optimizer.zero_grad()
        x = self.encoder_x(set_n)
        y_prime = self.encoder_x(set_a_counter)
        embeddings_x = torch.add(x, y_prime)
        embeddings_x = nn.functional.normalize(embeddings_x, dim=1)
        
        with torch.no_grad():
            self._momentum_update_encoder_c()
            x_prime = self.encoder_c(set_a)
            y = self.encoder_c(set_n_counter)
            embeddings_x_transform = torch.add(x_prime, y)
            embeddings_x_transform = nn.functional.normalize(embeddings_x_transform, dim=1)
        
        self.newEmbeddings = torch.stack((embeddings_x, embeddings_x_transform), dim=1).view(-1,embeddings_x_transform.shape[1])
        
        positive_pairs = torch.tensor([[i, i+1] for i in range(0,len(self.newEmbeddings),2)]).to(self.global_rank)
        alignment = utils.alignment_loss(embeddings=self.newEmbeddings, positive_pairs=positive_pairs)
        uniformity = utils.uniformity_loss(embeddings=self.newEmbeddings)
        
        loss = xent_loss(self.newEmbeddings,t=self.temperature)
        self.loss_metric.update(loss)

        self.alignment += alignment * len(train_x)
        self.uniformity += uniformity * len(train_x)
        self.total += len(train_x)
        
        self.log_dict(
            {
                'train_loss': self.loss_metric.compute(),
                'gross_loss': self.loss_metric_gross.compute(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'step': float(self.current_epoch),
            },
            on_step = True,
            on_epoch = False,
            prog_bar = True,
            sync_dist=True,
        )
        
        if(batch_idx % 100 == 0):
            y1 = train_x[:4]
            y2 = train_x_transform[:4]
            grid_x = torchvision.utils.make_grid(y1.view(-1, self.channel, self.image_size, self.image_size))
            grid_y = torchvision.utils.make_grid(y2.view(-1, self.channel, self.image_size, self.image_size))
            self.logger.experiment.add_image(self.dataset+"_N", grid_x, self.current_epoch)
            self.logger.experiment.add_image(self.dataset+"_A", grid_y, self.current_epoch)
        
        return loss
    

    def on_train_epoch_end(self):
        self.log_dict(
            {
                'alignment': self.alignment/self.total,
                'uniformity': self.uniformity/self.total,
                'step': float(self.current_epoch),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.loss_value.append(self.loss_metric.compute().cpu().numpy())
        self.loss_metric.reset()
        self.accuracy.reset()
        self.alignment = 0.0
        self.uniformity = 0.0
        self.total = 0
        
        #save model .. 
        if((self.current_epoch + 1) % 100 == 0):
            #self.gamma = self.gamma * 0.1
            save_path = os.path.join(self.save_path, self.checkpoint_dir)

            if(not os.path.exists(save_path)):
                os.makedirs(save_path)
                print("Path created...")
            file_path = os.path.join(save_path, "model" + str(self.current_epoch + 1) + ".tar")
            self.Save(file_path)
    
        if((self.current_epoch + 1) % 10 == 0):
            top1 = self.knn_monitor.test(net = deepcopy(self.encoder_x[0]), dataset_name=self.dataset)

            self.log_dict(
                {
                    'Knn Top-1': top1,
                    'step': float(self.current_epoch),
                    #'Knn Top-5': top5,
                },
                on_step = False,
                on_epoch = True,
                prog_bar = True,
                sync_dist=True,
            )
    
    def configure_optimizers(self):
        self.optimizer = optim.SGD([{'params': self.parameters(), 'lr': self.lr, 'momentum': 0.9, 'weight_decay': self.weight_decay}])
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.warmup_cosine_decay_leraning_rate)
        
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",  # Step the scheduler every epoch
                "frequency": 1,      # Step every 1 epoch
                "monitor": "train_loss",  # Monitor value for advanced schedulers
            },
        }
        
    # warmup + cosine decay
    def warmup_cosine_decay_leraning_rate(self, current_epoch):
        if current_epoch < self.warmup_epochs:
            # Linear warmup: Increase from warmup_lr to base_lr
            return self.warmup_lr + (self.lr - self.warmup_lr) * (current_epoch / self.warmup_epochs) / self.lr
        else:
            # Cosine decay
            decay_epochs = self.epochs - self.warmup_epochs
            decay_epoch = current_epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_epoch / decay_epochs))
            return cosine_decay

    """
    Function to save Our Model
    """
     
    def Save(self, model_path):
        
        encoder_x_state_dict = self.encoder_x.state_dict()
        encoder_c_state_dict = self.encoder_c.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

        torch.save({"encoder_x": encoder_x_state_dict,
                    "encoder_c": encoder_c_state_dict,
                    "optimizer": optimizer_state_dict},
                    model_path)
        print("model saved succesfully ..")
        
