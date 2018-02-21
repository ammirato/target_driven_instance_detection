# Target Driven Instance Detection

This is an implementation of the technique described in "Target Driven Instance Detection". It is written in python for use with Pytorch. 


## External Requirements
* Python 2 (might work with Python 3)
* [PyTorch](http://pytorch.org/)
* AVD THINGS

## Installation
***ADD SOMETHING FOR AVD CODE***

0. Make sure you have Pytorch (and torchvision)

1. Get the code
```
git clone https://github.com/ammirato/target_driven_instance_detection.git
```

2. Install the other requirements
```
cd target_driven_instance_detection/
pip install -r requirements.txt
```

3. Build the cython code for anchor boxes and non-max supression
```
cd model_defs/
./make.sh
```

4. Build the coco evaluation cython code 
```
cd ../evaluation/cocoapi/PythonAPI/
make all
```

5. Start training!
```
cd ../../../
python train_tdid.py
```



### TODO
1. Make ```proposal_layer``` and  ```anchor_target_layer``` cleaner, easier to understand, etc. 
2. is regression target correct on dummy gt box?
3. Hard negative mining on proposed boxes?
4. Add to README for AVD code
5. Add to README for extra data (target images, synthetic data)
6. Fix eval by object
7. Test det4class
8. provide trained models
9. make a note about downloading pretrained pytorch models

### Acknowledgements
This code started as a modification of a Faster-RCNN Pytorch implementation [here](https://github.com/longcw/faster_rcnn_pytorch), and still uses some of that code. (In particular nothing was changed in the nms code).



