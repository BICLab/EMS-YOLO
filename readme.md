Deep Directly-Trained Spiking Neural Networks for Object Detection

(1) Setup
This code has been tested with pytorch=1.10.1,py3.8, cuda11.3, cudnn8.2.0_0
The conda environment can be copied directly via environment.yml.
Some additional dependencies can be found in the requirements.txt.

(2)Files introduction
The datasets can be downloaded from their official websites.

The snn-resnet folder is used for object detection of COCO dataset, where the training file is train.py; 
the inference file is detect.py.  The model files are the same as in g1-resnet and can be copied directly.

The g1-resnet folder is used for object detection of Gen1 dataset.

The difference between the processing of these two datasets is that the data are processed differently, 
and also the data used for each time step is different.

(3)Traning
For gen1 dataset:
python path/to/train_g1.py --weights ***.pt --img 640
For coco dataset:
python train.py

(4)Inference
For gen1 dataset:
python val.py

For the coco dataset:
python detect.py

(5)Calculating the spiking rate
Dependencies can be downloaded from https://github.com/luo3300612/Visualizer
python calculate_fr.py