# iWildCam 2019

The aim of this project is to categorise animals from images provided in [iWildCam 2019 - FGVC6](https://www.kaggle.com/c/iwildcam-2019-fgvc6/overview) Kaggle dataset.

In this priject I used transfer learning.

I'm comparing 3 approaches, where all images resized to 100x100
- with some augmentation, all images used
- more augmentation and calculated class weights to reduce impact of imbalanced data
- only 20% of `empty` images used

**TODO**
- Try different models
- Two step iteration with image segmentation
