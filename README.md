# SSL framework for MedMnist
This is an official implementations of the paper "A self supervised learning framework for imbalanced medical imaging datasets".

## Requirements

    Python              - 3.10.12 
    Tensorboard         - 2.13.0  
    Pytorch             - 1.12.0+cu116 
    Pytorch-lightning   - 1.9.5 

## config settings
    config

        config.yaml

        backbone
            resnet18.yaml
            resnet50.yaml

        dataset
            PathMnist.yaml
            BloodMnist.yaml
            OCTMnist.yaml
            BreastMnist.yaml
            PneumoniaMnist.yaml
            OrganAMnist.yaml
            OrganCMnist.yaml
            OrganSMnist.yaml
            RetinaMnist.yaml
            TissueMnist.yaml
            DermaMnist.yaml

        model
            MedMnist_SSLModel.py

        post_training
            linear_evaluation.yaml

        training
            defauly.yaml
            pretraining.yaml

## How to Run

### (Pretraining & linear_evaluation)
    
    python3 main.py dataset.data_dir="path_to_dataset" dataset.save_path="path_to_save_model_on_each_nth_epoch" 
    Note - Default setting are - Dataset - DermaMnist, Model - MedMnist_SSL, Backbone - resnet50
 
#### or 

    As per the config files hierarchy use command line arguments to use more parameters

#### or

    Make changes in respective config file and then run - python3 main.py

## To access the tensorboard logs

    tensorboard --logdir results/pretrain_logs/
    tensorboard --logdir results/linear_eval_logs/

