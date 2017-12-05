## PyTorch Implementation of the Coupled GAN algorithm for Unsupervised Image-to-Image Translation

### License

Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

### Paper

[Ming-Yu Liu, Thomas Breuel, Jan Kautz, "Unsupervised Image-to-Image Translation Networks" NIPS 2017 Spotlight, arXiv:1703.00848 2017](https://arxiv.org/abs/1703.00848)

#### Two Minute Paper Summary
[![](./docs/two-minute-paper.png)](https://youtu.be/dqxqbvyOnMY) (We thank the Two Minute Papers channel for summarizing our work.)

#### The Shared Latent Space Assumption
[![](./docs/shared-latent-space.png)](https://www.youtube.com/watch?v=nlyXoX2aIek)

#### Result Videos

More image results are available in the [Google Photo Album](https://photos.app.goo.gl/5x7oIifLh2BVJemb2).

*Left: input.* **Right: neural network generated.** Resolution: 640x480

![](./docs/snowy2summery.gif)

*Left: input.* **Right: neural network generated.** Resolution: 640x480

![](./docs/day2night.gif)
![](./docs/dog_breed.gif)
![](./docs/cat_species.gif)

- [Snowy2Summery-01](https://youtu.be/9VC0c3pndbI)
- [Snowy2Summery-02](https://youtu.be/eUBiiBS1mj0)
- [Day2Night-01](https://youtu.be/Z_Rxf0TfBJE)
- [Day2Night-02](https://youtu.be/mmj3iRIQw1k)
- [Translation Between 5 dog breeds](https://youtu.be/3a6Jc7PabB4)
- [Translation Between 6 cat species](https://youtu.be/Bwq7BmQ1Vbc)

#### Street Scene Image Translation
From the first row to the fourth row, we show example results on day to night, sunny to rainy, summery to snowy, and real to synthetic image translation (two directions). 

For each image pair, *left is the input image*; **right is the machine generated image.**

![](./docs/street_scene.png)

#### Dog Breed Image Translation

![](./docs/dog_trans.png)

#### Cat Species Image Translation

![](./docs/cat_trans.png)

#### Attribute-based Face Image Translation

![](./docs/faces.png)

### Code usage

Please go to the [user manual page](USAGE.md)


