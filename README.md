# Make-Sketch-dataset
* MAS
* makeing sketch dataset

## Install dependencies

```
# You can use the library already exists on the computer
# I will only remind you to install special libraries

# most important
git clone https://github.com/BachiLi/diffvg
git submodule update --init --recursive
cd diffvg
python setup.py install

# If you have completed the above steps, Victory is within grasp
```

## Clone the repo
```
git clone https://github.com/hankunbo/Make-Sketch-dataset.git
cd Make-Sketch-dataset
```
## download mask parameter 
```
#download mask parameter from
https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ
#and put it under U2Net_/saved_models
```
## dataset
put your img dataset in your-imgpath 

## train
```
python main1.py --img_paths your-imgpath 
```
