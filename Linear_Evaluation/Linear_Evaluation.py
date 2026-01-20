import pytorch_lightning as pl
#import config
from Linear_Evaluation.Model_LE import linearlayer_training
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os

from data.RetinaMnist.retineMnist_dataset_le import RetinaMnist_DataModule_le
from data.BreastMnist.breastMnist_dataset_le import BreastMnist_DataModule_le
from data.PneumoniaMnist.pneumoniaMnist_dataset_le import PneumoniaMnist_DataModule_le
from data.BloodMnist.bloodMnist_dataset_le import BloodMnist_DataModule_le
from data.DermaMnist.dermaMnist_dataset_le import DermaMnist_DataModule_le
from data.PathMnist.pathMnist_dataset_le import PathMnist_DataModule_le
from data.PathMnist.pathMnist_dataset_le import PathMnist_DataModule_le
from data.TissueMnist.tissueMnist_dataset_le import TissueMnist_DataModule_le
from data.OrganCMnist.organCMnist_dataset_le import OrganCMnist_DataModule_le
from data.OrganSMnist.organSMnist_dataset_le import OrganSMnist_DataModule_le
from data.OrganAMnist.organAMnist_dataset_le import OrganAMnist_DataModule_le
from data.OCTMnist.octMnist_dataset_le import OCTMnist_DataModule_le

import utils

def Get_Model(model_name):
   
    model_function = model_name + "Model_LE"

    exec(f"generated_model = {model_function}", globals())
    return globals()['generated_model']
    

def Get_Dataset(dataset_name):
    dataset_function = dataset_name + "_DataModule_le"
    exec(f"generated_dataset = {dataset_function}", globals())
    return globals()['generated_dataset']

def Linear_Evaluation(trained_model, config):
     
    result_folder = utils.GetTensorboardDir(config, train_mode="linear_eval")
    linear_checkpoint_path = utils.GetCheckpointDir(config, train_mode= "linear_eval")
    #linear_checkpoint_path = os.path.join("results", config.dataset.name + "_linear", config.feature.mode, config.imbalance.imb_type)
    logger = TensorBoardLogger(result_folder, name = config.backbone.name)
    
    pretrained_filename = os.path.join(linear_checkpoint_path, (config.model.name + "ModelLE.ckpt"))
    checkpoint_callback = ModelCheckpoint(dirpath=pretrained_filename,
                                                 #save_weights_only=True,
                                                 mode = "min",
                                                 monitor='linear_evaluation_loss')
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    model = linearlayer_training(config)
    generated_dataset = Get_Dataset(config.dataset.name)
    dm = generated_dataset(trained_model, config)
    

    trainer = pl.Trainer(
        default_root_dir=os.path.join(linear_checkpoint_path, (config.model.name + "ModelLE")),
        logger = logger,
        accelerator='gpu',
        devices = config.post_training.devices,
        min_epochs = 1,
        max_epochs=config.post_training.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1,
    )
    '''
    if(os.path.exists(pretrained_filename)):
        print("Linear Layer Loading ...")
        #saved_linearlayer = "epoch=" + str(config.checkpoint_ll) + "-step=" + str(config.checkpoint_ll) + ".ckpt"
        saved_linearlayer = "epoch=" + str(config.post_training.checkpoint_toload) + "-step=" + str(config.post_training.checkpoint_toload) +".ckpt"
        #model = OurModel_LE.load_from_checkpoint(pretrained_filename)
        trainer.fit(model, dm, ckpt_path=os.path.join(pretrained_filename, saved_linearlayer))
    else:
        #pl.seed_everything(42)
        trainer.fit(model, dm)
        #trainer.validate(model, dm) 
    '''
    trainer.fit(model, dm)
    trainer.test(model, dm)
