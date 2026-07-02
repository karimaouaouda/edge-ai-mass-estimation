# DEPLOYMENT PIPELINE


## Project Pipeline
the project is a cascade models that work together in orger to estimate the mass of the object in the image, an image will go through these stages in order to get the mass in the final :
1. **Detection/Segmentation Stage** the image will be passed to the segmentation model based on [Yolo26m-seg](https://docs.ultralytics.com/models/yolo26), which will segment the waste objects in the image and return the segmented image. here we gonna extract the geometric properties of the segmented object, like the area and the perimeter, and we will use these properties to estimate the mass of the object in the next stage.

2. **Depth Stage** the segmented image will be passed to the depth model, we use [DepthAnything](https://depthanything.org/) or it's [official github repo](https://github.com/DepthAnything/Depth-Anything-V2) as a primary choice, here we gonna extract some physical properties of the segmented object, like the depth and the volume, and we will use these properties to estimate the mass of the object in the next stage.
    - `note:` we can use other depth models like [MiDaS](https://pytorch.org/hub/intelisl_midas_v2/) as a fallback choice if depth anything model is not working well.

3. **Mass Estimation Stage** the physical and geometric properties of the segmented object will be passed to the mass estimation model, which will estimate the mass of the object based on these properties. (for more details on the mass model training pipeline, refer to [Mass Estimation Model Training Pipeline](train-pipeline-mass.md)).

## DEPLOYABLE ASSETS

in order to make our project a real world project that can be demonstrated, we deployed : 
- **full pipeline models and scripts:** we deployed the full pipeline models, and we deploy a full orchestration script that will take an image as input and pass it through the full pipeline and return the mass of the object, the orchestration script will take care of all the things needed like : inference, connection to server, update models, ... . it's an agent that take care of everything.
    - **Overview:** the agent is deployed as a built wheel file, and it can be run on any machine that has python installed, and it can be run as a service in the background, and it can be connected to the dashboard to send logs and receive commands. this will be improved to docker image as we already use docker in the dashboard and databases.

- **Dashboard:** we deployed a dashboard that let us connect with the agent in the edge device, we can send commands from this dashboard to agent, see logs, update, health of edge device.
    - **Dev Stack:** we use Laravel as a full stack framework to build the dashboard.

- **Data Storage:** we deployed a data storage solution to store the results of the pipeline stages, including the estimated masses and the input images.
    - **Dev Stack:** we use MySQL as a database to store the results.
- **MQTT Broker:** we deployed an MQTT broker to facilitate communication between the edge device and the dashboard, allowing for real-time updates and command execution.
    - **Dev Stack:** we use Mosquitto as the MQTT broker. deployed on docker container, and we use the MQTT protocol to communicate between the edge device and the dashboard (exchange data and commands).


### EDGE DEVICE INFORMATION

we use [Nvidia Jetson Nano](https://www.nvidia.com/fr-fr/autonomous-machines/embedded-systems/jetson-nano/product-development/) to deploy the full pipeline, it's a small computer that can run deep learning models, and it has a GPU that can accelerate the inference of the models. we use it as an edge device to run the full pipeline and send the results to the dashboard. it's considered as the best choice for industrial edge devices, because it's small, cheap, and powerful enough to run deep learning models (built for AI).

for more information about edge computing and the importance of jetson nano in edge computing, refer to [nvidia edge computing and ai](https://www.nvidia.com/fr-fr/edge-computing/).