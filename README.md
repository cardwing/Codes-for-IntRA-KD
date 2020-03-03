Codes for "Inter-Region Affinity Distillation for Road Marking Segmentation"

## Requirements
- [PyTorch 0.3.0](https://pytorch.org/get-started/previous-versions/).
- Opencv
- cvbase

## Before start

Please follow [list](./list) to put ApolloScape in the desired folder. We'll call the directory that you cloned Codes-for-IntRA-KD as `$IntRA_KD_ROOT .

## Testing

1. Obtain model predictions from trained weights:
```
    cd $IntRA_KD_ROOT
    sh test_pspnet_multi_scale.sh
```

The output predictions will be saved to ```road05_new``` by default.

2. Transfer TrainID to ID:
```
    python road_npy2img_multi_process.py
```

The outputs will be stored in ```road05``` by default. 

3. Generate zip files:
```
    mkdir test
    mv road05 test/
    zip -r test.zip test
```

Now, just upload test.zip to [ApolloScape online server](http://apolloscape.auto/submit.html).(The top entity dubbed "Simple_train" in the leaderboard is the teacher model, i.e., ResNet-101)


4. (Optional) Produce color maps from model predictions:
```
    python trainId2color.py
```

5. (Optional) Leverage t-SNE to visualize the feature maps:

Please use the [script](https://github.com/cardwing/Codes-for-Steering-Control/blob/master/tools/draw_tsne.py) to perform the visualization.
 
## Training
```
    cd $IntRA_KD_ROOT
    sh train_pspnet_multi_scale.sh
```

## Citation

If you use the codes, please cite the following publication:
```
 
```

## Acknowledgement
This repo is built upon [ERFNet-CULane-PyTorch](https://github.com/cardwing/Codes-for-Lane-Detection).

## Contact
If you have any problems in reproducing the results, just raise an issue in this repo.

## To-Do List
- [ ] Training codes of IntRA-KD and various baseline KD methods for ApolloScape
