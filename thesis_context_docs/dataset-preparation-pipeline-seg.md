# DATASETS Preparation For my project

## DATASETS INFORMATION

- **Datasets Included**: the datasets that are included in the training pipeline are:
    - **TACO (Trash Annotations in Context ):** is an open image dataset of waste in the wild. It contains photos of litter taken under diverse environments, from tropical beaches to London streets.
        - `link:` [click here](https://tacodataset.org/) to access the dataset official site.

    - **AquaTrash:**This dataset contains 369 images of Trash used for deep learning. Each image is manually labelled by our team for accurate detections making a total of 470 bounding boxes. There are total 4 classes {(0: glass), (1:paper), (2:metal), (3:plastic)}
        - `link:` [click here](https://www.kaggle.com/datasets/harshpanwar/aquatrash) to access the dataset details in kaggle
        - `link:` [click here](https://www.sciencedirect.com/science/article/pii/S2666016420300244?via%3Dihub) to access the official Paper that introduces the dataset. (_this is important link_)

    - **RealWaste:** An image classification dataset of waste items across 9 major material types, collected within an authentic landfill environment.

        - `link:` [click here](https://www.kaggle.com/datasets/joebeachcapital/realwaste) to access the dataset details in kaggle.
        - `link:` [click here](https://www.mdpi.com/2078-2489/14/12/633) to access the official Paper that introduces the dataset. (_this is important link_)
        - `link:` [click here](https://archive.ics.uci.edu/dataset/908/realwaste) to access additional paper that introduces the dataset.


## DATASETS PREPARATION PIPELINE

### the model will be used : 

in this project the first stage is the **Segmentation** stage, where the image will be provided to a segmentation model **YOLO26m-SEG** in our case, where will be used to detect the waste in the image and segment it.

### DATASETS NATURE

depends on what we want to do, the datasets must contain: \
    - **Segmentation Annotations**: if we want to train a segmentation model, the datasets must contain segmentation annotations, which are usually in the form of masks or polygons that outline the objects of interest in the images.

    - **Same Classes That we Want to Detect**: if we want to train a segmentation model to detect specific classes of objects, the datasets must contain annotations for those classes. For example, if we want to detect waste in images, the datasets must contain annotations for waste objects.

### DATASETS LIMITATIONS

the datasets that we gonna use have limitations, are : 
- **TACO Dataset Limitations**: The TACO dataset is a relatively small dataset, which may limit the performance of the segmentation model. Additionally, the dataset may not contain enough examples of certain classes of waste, which could lead to poor performance on those classes. not same environment that we want to detect (not same environment that we gonna use the model in). it have segmentation annotations, which means that it can be used to train a segmentation model.

- **AquaTrash Dataset Limitations**: The AquaTrash dataset is a relatively small dataset, which may limit the performance of the segmentation model. Additionally, the dataset may not contain enough examples of certain classes of waste, which could lead to poor performance on those classes. and it have not segmentation annotations, which means that it cannot be used to train a segmentation model. not same environment that we want to detect (not same environment that we gonna use the model in).

- **RealWaste Dataset Limitations**: The RealWaste dataset is a good dataset in size, how ever it have not segmentation annotations, which means that it cannot be used to train a segmentation model.

### LIMITATIONS SOLUTIONS

- **TACO LIMITATIONS:** as for taco, the only limitation is the environment, because we have no control over the environment in which the images were taken. the size of the dataset is also a limitation, but it can be mitigated by merging datasets.

- **AquaTrash LIMITATIONS:** as for AquaTrash, the only limitation that we can not control is the environment, because we have no control over the environment in which the images were taken. the size of the dataset is also a limitation, but it can be mitigated by merging datasets. and it have not segmentation annotations, which means that it cannot be used to train a segmentation model. how ever the segmentation annotations can be solved by annotate the dataset, but it will take a lot of time and effort.

- **RealWaste LIMITATIONS:** the real_waste dataset in other hand have a good environment, but it have not segmentation annotations, which means that it cannot be used to train a segmentation model. how ever the segmentation annotations can be solved by annotate the dataset, but it will take a lot of time and effort.


### Step Taken

In order to overcome the limitations of the datasets, we have taken the following steps:
- **Annotate the datasets**: We have annotated the datasets to include segmentation annotations, which allows us to train a segmentation model. this process take this steps :
    - we manually annotate the ``AquaTrash dataset`` to include segmentation annotations.
    - we train a detection model on `taco dataset` and `AquaTrash dataset` as they have bounding boxes annotations, and we use this model to detect the waste in the images of the `RealWaste dataset`, and we use the detected bounding boxes for the `RealWaste dataset`

    - We run a segmentation Process On the `RealWaste` dataset, where we use a segmentation model [`SegmentAnything`](https://segment-anything.metademolab.com/) to segment the waste in the images of the `RealWaste dataset`, `SegmentAnything` is a model that can segment any object in an image and can be use Bounding boxes to segment the objects in the image more precisely. click [here](https://segment-anything.metademolab.com/) to access the official site of the model.

    - the resulted Segmentation annotations from the `SegmentAnything` model was not perfect for all the images, so we manually correct the segmentation annotations for the images that were not segmented correctly by the `SegmentAnything` model using a tool called [CVAT](https://www.cvat.ai/), click [here](https://www.cvat.ai/) to access the official site of the tool. we use premium version of the tool (33$/month) to be able to use Advanced features of the tool.

    so for Segmentation Task in the project we Successfully Annotate two full datasets with proper Segmentation Annotations, which are `AquaTrash` and `RealWaste` datasets, with a Sum of 369 images for `AquaTrash` dataset and 4752 images for `RealWaste`. in total : `4752 + 369 = 5 121 images` which is a good number of images as a start for training a segmentation model.



`futur improvements:`
    - we can use trashnet too
    - instead of train the model on all of merged datasets, we can pretrain the model on `taco` and `AquaTrash` datasets (bad environment for us, but usefull segs), and then fine-tune the model on `RealWaste` and `trashnet` datasets, this will improve the performance of the model as it will learn more features from the pretraining datasets.


- `very important note:` `trashnet` dataset is not used in this project, how ever we annotate it, it have more then 2500 image without annotations, and we annotate it to include segmentation annotations, which allows us to train a segmentation model. the process is same as annotating `RealWaste`dataset.
    - `official dataset repo:` [click here](https://github.com/garythung/trashnet) to access the dataset details in github repository.
    - `official dataset kaggle:` [click here](https://www.kaggle.com/datasets/feyzazkefe/trashnet) to access the dataset details in kaggle.
this make the total annotated images is : `5 121 + 2 527= 7 648 images` which is a good number of images as a start for training a segmentation model.