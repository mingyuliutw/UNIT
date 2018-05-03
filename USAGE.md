[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
## UNIT: UNsupervised Image-to-image Translation Networks

### License

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

### Dependency


pytorch, yaml, tensorboard (from https://github.com/dmlc/tensorboard), and tensorboardX (from https://github.com/lanpa/tensorboard-pytorch).


The code base was developed using Anaconda with the following packages.
```
conda install pytorch torchvision cuda80 -c soumith
conda install -y -c anaconda pip; 
conda install -y -c anaconda yaml;
pip install tensorboard tensorboardX;
```

We also provide a [Dockerfile](Dockerfile) for building an environment for running the UNIT code.

### Example Usage

#### Testing 

First, download the [pretrained models](https://drive.google.com/open?id=1R9MH_p8tDmUsIAjKCu-jgoilWgANfObx) and put them in `models` folder.

Run the following command to translate GTA5 images to Cityscape images
    
    python test.py --trainer UNIT --config configs/unit_gta2city_list.yaml --input inputs/gta_example.jpg --output_folder outputs/gta2city --checkpoint models/unit_gta2city.pt --a2b 1
    
Run the following command to translate Cityscape images to GTA5 images
    
    python test.py --trainer UNIT --config configs/unit_gta2city_list.yaml --input inputs/city_example.jpg --output_folder outputs/city2gta --checkpoint models/unit_gta2city.pt --a2b 0    
 
#### Training

1. Download the dataset you want to use. For example, you can use the GTA5 dataset provided by [Richter et al.](https://download.visinf.tu-darmstadt.de/data/from_games/) and Cityscape dataset provided by [Cordts et al.](https://www.cityscapes-dataset.com/).

3. Setup the yaml file. Check out `configs/unit_gta2city_folder.yaml` for folder-based dataset organization. Change the `data_root` field to the path of your downloaded dataset. For list-based dataset organization, check out `configs/unit_gta2city_list.yaml`

3. Start training
    ```
    python train.py --trainer UNIT --config configs/unit_gta2city_folder.yaml
    ```
    
4. Intermediate image outputs and model binary files are stored in `outputs/unit_gta2city_folder`

##### Notes

Note that for learning to translate images at 512x512 resolution, you would need a GPU card with 16GB memory such as Tesla V100. For 256x256 resolution, a GPU with 12GB memory should be sufficient.