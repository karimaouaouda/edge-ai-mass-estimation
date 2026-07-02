# DATASETS Preparation For my project

## DATASETS INFORMATION

in order to train the mass estimation model, we need to prepare the datasets that will be used for training the model. unfortunatly, there is no dataset that contains the mass of the waste in the images, so we need to prepare our own dataset.

so we manually take pics of the waste in the landfill, and we manually measure the mass of the waste in the images, and we manually annotate the images with the mass of the waste in the images and other information that will be used for training the model, such as class of the waste, dimensions. we also manually annotate the images with the segmentation annotations as we need physical informations to include on the mass model, we use [CVAT](https://www.cvat.ai/) to annotate the images, click [here](https://www.cvat.ai/) to access the official site of the tool. we use premium version of the tool (33$/month) to be able to use Advanced features of the tool.

`important note:` We develope custom tool to help us to annotate the images with the mass of the waste in the images, and there dimensions, this tool was developed from scratch using FastAPI.

the process of building the dataset is determined by the following steps:
1. we manually take pics of the waste in the landfill, and we manually measure the mass and dimensions and use our custom tool to do that.
2. we take the images and annotate them in [CVAT](https://www.cvat.ai/) with Segmentation annotations.
3. we merge the image annotations json with the mass and dimensions annotations json to have a single json file that contains all the initial informations that we need in first.
4. we calculate the important features that we need to train the model, such as area, perimeter, and other features that will be used for training the model.



``IMPORTANT NOTE:`` as we have no time to build a large dataset, we used a pseudo dataset that contains images from `RealWsate` dataset, and we estimate the mass of the waste images using LargeLanguage Models.