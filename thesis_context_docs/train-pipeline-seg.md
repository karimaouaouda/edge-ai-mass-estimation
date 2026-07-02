# TRAIN PIPELINE FOR SEGMENTATION MODEL

## DATASETS PREPARATION

in order to train a segmentation model, you need to prepare your datasets. This involves collecting and organizing your images and their corresponding segmentation masks.

refer to : [DATASETS Preparation For my project](dataset-preparation-pipeline-seg.md) for more information about how we prepared the datasets for training the segmentation model.



## TRAINING PIPELINE

u can analyze the training pipeline for the segmentation model in this project. this include preparation, optimization using optuna, and training the model using the prepared datasets. (should mention sources of important technologies)

some of the important technologies used in this project are:
- `optuna:` for hyperparameter optimization, we used optuna library to optimize the hyperparameters of the model, such as learning rate, batch size, and number of epochs. we used optuna to find the best hyperparameters for our model, and we used the best hyperparameters to train the model.
    - `official site:` refer to [optuna](https://optuna.org/) for more information about the library.

    - `huggingface pipelines / zenml`: we used huggingface pipelines and zenml to create a pipeline for training and inference for all models. (read the prject for get the real and full information about the pipeline)
        - `zenml offical site:` refer to [zenml](https://www.zenml.io/) for more information about the library.

as we have not a powerful machine to train the model, we used [Kaggle](https://www.kaggle.com/) platform to train the model, as it provides free GPU and TPU resources for training deep learning models. we used the free version of the platform, which provides 30h of GPU and 30h of TPU per week, and we used the GPU resources to train the model. we need GPU because we are training a Computer Vision model, so we need GPU to accelerate the training process.


we upload the notebook as an entry point for the training, where we upload the script as an utility script for kaggle kernel.

## TRAINING MODEL
we use the following model to train the segmentation model:
- `YOLO26m-seg` model from ultralytics, have a strong fast architecture for segmentation tasks, and it is easy to use and customize. we used the pre-trained weights of the model on COCO dataset, and we fine-tuned the model on our dataset. for more information about the model, refer to [YOLOv26](https://docs.ultralytics.com/models/yolo26).