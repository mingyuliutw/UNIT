[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)

## UNIT: UNsupervised Image-to-image Translation Networks

### License

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

### Paper

[Ming-Yu Liu, Thomas Breuel, Jan Kautz, "Unsupervised Image-to-Image Translation Networks" NIPS 2017](https://arxiv.org/abs/1703.00848)

Please cite our paper if this software is used in your publications.

### Dependency


pytorch, yaml, opencv, and tensorboard (from https://github.com/dmlc/tensorboard).


The code base was developed using Python 2 in Anaconda2 with the following packages.
```
conda install pytorch torchvision cuda80 -c soumith
conda install -y -c anaconda pip; 
conda install -y -c anaconda yaml;
conda install -y -c menpo opencv;
pip install tensorboard==1.0.0a6;
```

We also provide a [Dockerfile](Dockerfile) for building an environment for running the UNIT code.

### Example Usage

#### Testing 

###### Cat to Tiger Translation
1. Download the pretrained model in [link](https://drive.google.com/open?id=0BwpOatrZwxK6V1Bwai1GZFQ2Q0k) to <outputs/unit/cat2tiger>

2. Go to <src> and run the following command to translate cats to tigers
    ```
    python cocogan_translate_one_image.py --config ../exps/unit/cat2tiger.yaml --a2b 1 --weights ../outputs/unit/cat2tiger/cat2tiger_gen_00500000.pkl --image_name ../images/cat001.jpg --output_image_name ../results/cat2tiger_cat001.jpg
    ```
    ```
    python cocogan_translate_one_image.py --config ../exps/unit/cat2tiger.yaml --a2b 1 --weights ../outputs/unit/cat2tiger/cat2tiger_gen_00500000.pkl --image_name ../images/cat002.jpg --output_image_name ../results/cat2tiger_cat002.jpg
    ```

4. Check out the results in <results>. Left: Input. Right: Output
 - ![](./results/cat2tiger_cat001.jpg)
 - ![](./results/cat2tiger_cat002.jpg)
 
###### Corgi to Husky Translation
1. Download the pretrained model in [link](https://drive.google.com/open?id=0BwpOatrZwxK6NktUSWZRNE14Ym8) to <outputs/unit/corgi2husky>

2. Go to <src> and run the following command to translate corgis to huskies
    ```
    python cocogan_translate_one_image.py --config ../exps/unit/corgi2husky.yaml --a2b 1 --weights ../outputs/unit/corgi2husky/corgi2husky_gen_00500000.pkl --image_name ../images/corgi001.jpg --output_image_name ../results/corgi2husky_corgi001.jpg
    ```
    ```
    python cocogan_translate_one_image.py --config ../exps/unit/corgi2husky.yaml --a2b 0 --weights ../outputs/unit/corgi2husky/corgi2husky_gen_00500000.pkl --image_name ../images/husky001.jpg --output_image_name ../results/husky2corgi_husky001.jpg
    ```

3. Check out the results in <results>. Left: Input. Right: Output
 - ![](./results/corgi2husky_corgi001.jpg)
 - ![](./results/husky2corgi_husky001.jpg)
 
###### Synthetic-to-real and Real-to-synthetic Translation
1. Download the pretrained model in [link](https://drive.google.com/open?id=1iTQFpyMmMLPe1eY2q-7g-3b5OSlYqc6X) to <outputs/unit/street_scene>

2. Go to <src> and run the following command to translate a real image to a synthetic image.
    ```
    python cocogan_translate_one_image.py --config ../exps/unit_local/synthia2cityscape.yaml --a2b 0 --weights ../outputs/unit/street_scene/synthia2cityscape_gen_00250000.pkl --image_name ../images/freiburg_000000_000021_leftImg8bit.png --output_image_name ../results/synthetic_freiburg_000000_000021_leftImg8bit.png
    ```
    
3. Check out the results in <results>. Left: Input. Right: Output
 - ![](./results/synthetic_freiburg_000000_000021_leftImg8bit.png)

#### Training
1. Download the aligned and crop version of the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to <datasets/celeba>. 

2. Go to <datasets/celeba> and crop the middle region of the face images and resize them to 128x128
    ```
    python crop_and_resize.py;
    ```

3. Setup the yaml file. Check out <exps/unit/blondhair.yaml>

4. Go to <src> and do training
     ```
    python cocogan_train.py --config ../exps/unit/blondhair.yaml --log ../logs
    ```
5. Go to <src> and do resume training 
     ```
    python cocogan_train.py --config ../exps/unit/blondhair.yaml --log ../logs --resume 1
    ```
    
6. Intermediate image outputs and model binary files are in <outputs/unit/blondhair>

For more pretrained models, please check out the google drive folder [Pretrained models](https://drive.google.com/open?id=0BwpOatrZwxK6UGtheHgta1F5d28).
#### SVHN2MNIST Adaptation

1. Go to <src> and execute
     ```
    python cocogan_train_domain_adaptation.py --config ../exps/unit/svhn2mnist.yaml --log ../logs
    ```
