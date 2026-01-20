from torch import nn, optim
import pytorch_lightning as pl
import torch
import torchmetrics
import os
from torchmetrics.functional import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from torchmetrics.classification import MulticlassAUROC

from matplotlib.colors import ListedColormap

class linearlayer_training(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        #self.automatic_optimization = False
        self.model_name = config.model.name
        self.backbone = config.backbone.name
        self.linear_layer = nn.Linear(config.backbone.feature_size, config.dataset.num_classes) 
        self.dataset = config.dataset.name
        self.save_path = config.dataset.save_path
        self.batch_size = config.training.batch_size
        self.data_dir = config.dataset.data_dir
        self.checkpoint_dir = utils.GetCheckpointDir(config, train_mode="linear_eval")
        self.num_classes = config.dataset.num_classes
        self.imb_type = config.dataset.imbalance_type

        self.lr = config.post_training.lr
        self.epochs = config.post_training.max_epochs
        self.steps = config.post_training.steps
        self.weight_decay = config.post_training.weight_decay
        self.warmup_epochs = config.post_training.warmup_epochs
        self.warmup_lr = config.post_training.warmup_lr
        
        self.outputs = []
        self.val_outputs = []
        self.accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes = self.num_classes)
        #self.accuracy_top5 = torchmetrics.Accuracy(task = 'multiclass', num_classes = self.num_classes, top_k=5)
        self.auc = MulticlassAUROC(num_classes=self.num_classes,average='macro')
        self.loss_metric = torchmetrics.MeanMetric()
        self.criterion = nn.CrossEntropyLoss()
        
        self.class_names = [str(i) for i in range(self.num_classes)]
        self.train_rare_class = []
        self.train_freq_class = []
        self.y_test = []
        self.pred = []
        self.train_accuracy = []
        self.test_accuracy = []

    def forward(self, x):
        x = self.linear_layer(x)
        return x

    def on_train_epoch_start(self):
        self.y_test.clear()
        self.pred.clear()

    def training_step(self, batch, batch_idx):
        
        self.optimizer.zero_grad()

        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output, target)
        self.loss_metric.update(loss)
        _, predicted = torch.max(output.data, 1)
        self.y_test.extend(target.cpu())
        self.pred.extend(predicted.cpu())
        utils.adjust_learning_rate(self.optimizer, self.lr, self.current_epoch, self.epochs)
        self.log_dict(
            {
                'linear_evaluation_loss': self.loss_metric.compute(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'step': float(self.current_epoch),
            },
            on_step = True,
            on_epoch = False,
            prog_bar = True,
            sync_dist=True,
        )
        
        self.outputs.append({'output': output, 'label': target})        
        
        return loss
   
    def on_train_epoch_end(self):
        output = torch.cat([x["output"] for x in self.outputs])
        target = torch.cat([x["label"] for x in self.outputs])
        accuracy = self.accuracy(output, target)
        #accuracy_top5 = self.accuracy_top5(output, target)
    
        
        self.log_dict(
            {
                'Linear_Evaluation_Acc': accuracy,
                'step': float(self.current_epoch),
                #'Acc_Top5': accuracy_top5,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        #self.y_test.clear()
        #self.pred.clear()

        self.loss_metric.reset()
        self.accuracy.reset()
        #self.accuracy_top5.reset()
        self.outputs.clear()
    
    def log_confusion_matrix(self, preds, targets):
        # Compute confusion matrix
        cm = confusion_matrix(
            preds=preds,
            target=targets,
            num_classes=self.num_classes,
            task="multiclass",  # Specify task type
        )

        # Plot confusion matrix as heatmap
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            cm.cpu().numpy(),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        #plt.title("Confusion Matrix")

        # Log to TensorBoard
        self.logger.experiment.add_figure("Confusion Matrix", fig, global_step=self.current_epoch)
        save_path = "confusion_matrix" + "_" + self.model_name + "_" + self.dataset + "_" + self.imb_type + ".pdf"
        fig.savefig(save_path, dpi=600)
        #plt.savefig(self.save_path, dpi=300)
        plt.close(fig)

    def log_tsne(self, features, labels):
        plt.rcParams.update({
            "font.size": 20,
            "font.family": "serif",
            "axes.labelsize": 18,
            "axes.titlesize": 17,
            "legend.fontsize": 18,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "pdf.fonttype": 42,    # Editable text (Type 42)
            "ps.fonttype": 42
        })

        cmap = ListedColormap(plt.cm.tab10.colors[:self.num_classes])

        print("Applying t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
        features_2d = tsne.fit_transform(features.cpu().detach().numpy())
        labels = labels.cpu()
        print("Plotting t-SNE...")
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, s=60)
        cbar = plt.colorbar(scatter, label='Labels')
        # Force EXACT tick positions
        cbar.set_ticks(range(self.num_classes))

        # Force EXACT tick labels
        cbar.set_ticklabels([str(i) for i in range(self.num_classes)])

        #plt.title("2D t-SNE Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        save_path = "tsne_visualization" + "_" + self.model_name + "_" + self.dataset + "_" + self.imb_type + ".pdf"
        plt.savefig(save_path, dpi=600)
        print("t-SNE visualization saved to tsne_visualization.png")
        plt.close()

    
    def on_train_end(self):
        
        acc_per_class = torchmetrics.functional.accuracy(torch.Tensor(self.y_test), torch.Tensor(self.pred), task='multiclass',num_classes=self.num_classes,average = 'none')
        print(acc_per_class)

        self.y_test.clear()
        self.pred.clear()
    
    def test_step(self, batch, batch_idx): 
        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output, target)
        accuracy = self.accuracy(output, target)
        #accuracy_top5 = self.accuracy_top5(output, target)
        
        _, predicted = torch.max(output.data, 1)
        self.y_test.extend(target.cpu())
        self.pred.extend(predicted.cpu())
        
        self.log_dict(
            {
                'test_loss': loss,
                'test_accuracy': accuracy,
                #'test_accuracy_top5': accuracy_top5,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        
        self.outputs.append({'output': output, 'label': target})        

    def on_test_epoch_end(self):
        print("Evaluation Started...")
        if ("Mnist" in self.dataset):
            # Aggregate predictions and labels
            all_labels = torch.cat([x["label"] for x in self.outputs])
            all_preds = torch.cat([x["output"] for x in self.outputs])


            # t-SNE Visualization
            self.log_tsne(all_preds, all_labels)
            # Confusion Matrix
            self.log_confusion_matrix(all_preds, all_labels)
            self.outputs.clear()
        
            auc = self.auc(all_preds,all_labels)
            print("AUC" + str(auc))

        acc_per_class = torchmetrics.functional.accuracy(torch.Tensor(self.y_test), torch.Tensor(self.pred), task='multiclass',num_classes=self.num_classes,average = 'none')
        print(acc_per_class)
    def Save_LinearLayer(self):

        #save linear layer .. 
        save_path = os.path.join(self.save_path, self.checkpoint_dir)
        if(not os.path.exists(save_path)):
            os.makedirs(save_path)
            print("Path created...")
        file_path = os.path.join(save_path, "linear.tar")
        
        self.Save(file_path)
        
    
    """
    Function to save Our Model
    """
     
    def Save(self, linear_layer_path):
        
        linear_layer_state_dict = self.linear_layer.state_dict()

        torch.save({"linear_layer": linear_layer_state_dict},
                    linear_layer_path)
        print("Lienar layer saved succesfully ..")
    
    def configure_optimizers(self):
        self.optimizer = optim.AdamW([{'params': self.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay}])
        return self.optimizer
