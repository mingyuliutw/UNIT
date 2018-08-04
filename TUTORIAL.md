[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
## UNIT Tutorial

In this short tutorial, we will guide you through setting up the system environment for running the UNIT, which stands for unsupervised image-to-image translation, software and then show several usage examples.

### Background

Unsupervised image-to-image translation concerns learning an image translation model that can map an input image in the source domain to a corresponding image in the target domain without paired supervision on the mapping function. Typically, the training data consists of two datasets of images. One from each domain and paired of corresponding images between domains are unavailable. For example, to learning a summer-to-winter translation mapping function, the model only has access to a dataset of summer images and a dataset of winter images during training. 

### Algorithm

<img src="https://raw.githubusercontent.com/NVIDIA/UNIT/master/docs/shared-latent-space.png" width="800" title="Assumption"> 

The unsupervised image-to-image translation problem is an ill-posed problem. It basically aims at discovering the joint distribution from samples of marginal distributions. From the coupling theory in probability, we know there exists infinitely many possible joint distributions that can arrive to two given marginal distributions. To find the target solution, one would have to incorporate the right inductive bias. One has to use additional assumptions. UNIT is based on the shared-latent space assumption as illustrated in the figure above. Basically, it assumes that latent representations of a pair of corresponding images in two different domains share the same latent code. Although we do not have any pairs of corresponding images during training, we assume their existences and utilize network capacity constraint to encourage discovering the true joint distribution. 
 
 As shown in the figure above, UNIT consists of 4 networks, 2 from each domain 

1. source domain encoder (for extracting a domain-shared latent code for image in the source domain)
2. source domain decoder (for generating an image in the source domain using a latent code, either from the source or target domains)
3. target domain encoder (for extracting a domain-shared latent code for image in the target domain)
4. target domain decoder (for generating an image in the target domain using a latent code, either from the source or target domains)

In the test time, for translating images from the source domain to the target domain, it utilizes the source domain encoder to encoder the source domain image to a shared-latent code. It then utilizes the target domain decoder to generate an image in the target domain.

### Requirments

- Hardware: PC with NVIDIA Titan GPU. For large resolution images, you need NVIDIA Tesla P100 or V100 GPUs, which have 16GB+ GPU memory. 
- Software: *Ubuntu 16.04*, *CUDA 9.1*, *Anaconda3*, *pytorch 0.4.1*
- System package
  - `sudo apt-get install -y axel imagemagick` (Only used for demo)  
- Python package
  - `conda install pytorch=0.4.1 torchvision cuda91 -y -c pytorch`
  - `conda install -y -c anaconda pip`
  - `conda install -y -c anaconda pyyaml`
  - `pip install tensorboard tensorboardX`

### Docker Image

We also provide a [Dockerfile](Dockerfile) for building an environment for running the MUNIT code.

  1. Install docker-ce. Follow the instruction in the [Docker page](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1)
  2. Install nvidia-docker. Follow the instruction in the [NVIDIA-DOCKER README page](https://github.com/NVIDIA/nvidia-docker).
  3. Build the docker image `docker build -t your-docker-image:v1.0 .`
  4. Run an interactive session `docker run -v YOUR_PATH:YOUR_PATH --runtime=nvidia -i -t your-docker-image:v1.0 /bin/bash`
  5. `cd YOUR_PATH`
  6. Follow the rest of the tutorial.


### Training

We provide several training scripts as usage examples. They are located under `scripts` folder. 
- `bash scripts/unit_demo_train_edges2handbags.sh` to train a model for sketches of handbags to images of handbags translation.
- `bash scripts/unit_demo_train_edges2shoes.sh` to train a model for sketches of shoes to images of shoes translation.
- `bash scripts/unit_demo_train_summer2winter_yosemite256.sh` to train a model for Yosemite summer 256x256 images to Yosemite winter 256x256 image translation.

1. Download the dataset you want to use. For example, you can use the GTA5 dataset provided by [Richter et al.](https://download.visinf.tu-darmstadt.de/data/from_games/) and Cityscape dataset provided by [Cordts et al.](https://www.cityscapes-dataset.com/).

3. Setup the yaml file. Check out `configs/unit_gta2city_folder.yaml` for folder-based dataset organization. Change the `data_root` field to the path of your downloaded dataset. For list-based dataset organization, check out `configs/unit_gta2city_list.yaml`

3. Start training
    ```
    python train.py --trainer UNIT --config configs/unit_gta2city_folder.yaml
    ```
    
4. Intermediate image outputs and model binary files are stored in `outputs/unit_gta2city_folder`


### Testing

First, download our pretrained models for the gta2cityscape task and put them in `models` folder.

#### Pretrained models 

|  Dataset    | Model Link     |
|-------------|----------------|
| gta2cityscape |   [model](https://drive.google.com/open?id=1R9MH_p8tDmUsIAjKCu-jgoilWgANfObx) | 

#### Translation

First, download the [pretrained models](https://drive.google.com/open?id=1R9MH_p8tDmUsIAjKCu-jgoilWgANfObx) and put them in `models` folder.

Run the following command to translate GTA5 images to Cityscape images
    
    python test.py --trainer UNIT --config configs/unit_gta2city_list.yaml --input inputs/gta_example.jpg --output_folder results/gta2city --checkpoint models/unit_gta2city.pt --a2b 1
    
The results are stored in `results/gta2city` folder. You should see images like the following.    
    
| Input Photo | Output Photo |
|-------------|--------------|
| <img src="https://raw.githubusercontent.com/NVIDIA/UNIT/master/results/gta2city/input.jpg" width="384" title="Input"> | <img src="https://raw.githubusercontent.com/NVIDIA/UNIT/master/results/gta2city/output.jpg" width="384" title="Output"> |     
    
Run the following command to translate Cityscape images to GTA5 images
    
    python test.py --trainer UNIT --config configs/unit_gta2city_list.yaml --input inputs/city_example.jpg --output_folder results/city2gta --checkpoint models/unit_gta2city.pt --a2b 0    
 
 The results are stored in `results/city2gta` folder. You should see images like the following.
 
| Input Photo | Output Photo |
|-------------|--------------|
| <img src="https://raw.githubusercontent.com/NVIDIA/UNIT/master/results/city2gta/input.jpg" width="384" title="Input"> | <img src="https://raw.githubusercontent.com/NVIDIA/UNIT/master/results/city2gta/output.jpg" width="384" title="Output"> |
 
 