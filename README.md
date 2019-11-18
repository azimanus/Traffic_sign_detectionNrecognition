# Traffic_sign_detectionNrecognition

This Project is based on the tutorial: 
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

Here we have an overview of steps I followed:

(PS: I used Anaconda python distribution on ubuntu 16 .04)
I. Installation of requirements:
1. Tensorflow API
+ Create new conda environment:
conda create -n tensorflow_cpu pip python=3.6
activate tensorflow_cpu

+ install tensorflow:
pip install --ignore-installed --upgrade tensorflow==1.9

2. TensorFlow Models Installation
+ install prerequisits
$ conda install pillow, xml, jupyter, opencv, cython
the versions packages above used are:

+ create new folder with the name TensorFlow
(ex: /home/<user>/TensorFlow)
+ cd into TensorFlow directory
+ clone the model in side TensorFlow directory
$ git clone https://github.com/tensorflow/models.git

3. Protobuf installation/compilation:
+ Install protobuf using Homebrew:
$ brew install protobuf
+ adding environment variables:
$ cd <path_to_your_TensorFlow_directory>/models/research/
$ protoc object_detection/protos/*.proto –python_out=.
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

(!!! The commands above should be run whenever we open new terminal)
From <path_to_your_TensorFlow_directory>/models/research/ , run:
$ python setup.py build
$ python setup.py install
(!!! we run the two commands above whenever we change or update the object-detection python package)

4. LabelImg Installation
+ create new conda environment
$ conda create -n labelImg pyqt=5
$ activate labelImg
+ inside TensorFlow directory, we create new directory ‘addons’

TensorFlow
... addons
    ... labelImg
... models
    ... official
    ... research
    ... samples
    ... tutorials
    
+ clone labelImg repo
$ git clone https://github.com/tzutalin/labelImg.git
+ install dependencies:
$ sudo apt-get install pyqt5-dev-tools
$ sudo pip install -r requirements/requirements-linux-python3.txt
$ make qt5py3
+ test
$ cd TensorFlow\addons\labelImg
$ python labelImg.py

II. Training Custom Object Detection:
1. Workspace organization
+ create new folder “workspace” under TensorFlow directory, and another one “training-demo” under “workspace” just have been created:

... addons
    ... labelImg
... models
    ... official
    ... research
    ... samples
    ... tutorials
... workspace
    ... training_demo
    
+ under “training-demo”  folder we add sub-folders: 

... annotations
... images
    ... test
    ... train
... pre-trained-model
... training
... README.md

2. preparing annotations
here we use labelImg to annotate our own dataset. One we collect images for diffrent classes we start labeling operation
(I used 3 classes each one contains 100 samples, 90% for training and 10% for testing)
$ activate labelImg
$ python labelImg.py ..\..\workspace\training_demo\images
+ creating Label Map
a file that maps each label (class name) to an id [label_map.pbtxt]. It should be placed in
training-demo/annotations folder.
3. generate tf records
we add “scripts” and “scripts/preprocessing” folders

TensorFlow
... addons
   ... labelImg
... models
   ... official
   ... research
   ... samples
   ... tutorials
... scripts
   ... preprocessing
... workspace
    ... training_demo

+ converting *.xml to *.csv
- under ‘scripts/preprocessing’run xml_to_csv.py script   
$ python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

$ python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv

→ 2 files will be generated under ‘training-demo/annotations’ : test_label.csv and train_label.csv

+ Converting from *.csv to *.record
- under ‘scripts/preprocessing’generate_tfrecord.py script
$ python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv
--img_path=<PATH_TO_IMAGES_FOLDER>/train  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

$ python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv
--img_path=<PATH_TO_IMAGES_FOLDER>/test
--output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record

→ 2 files will be generated under ‘training-demo/annotations’ : test.record and train.record
4. configure training
+ download configuration file of the model choosed (in this case: ssd_mobilenet_v3_coco.config) from:
https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs

+ we need also to download the pre-trained NN
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models

+ we extract the content of *.tar.gz inside ‘training-demo/pre-trained-model’

+ follow comment of .config file and make the modifications
(.config file must be saved in ‘training-demo/training directory’)

5. train the model
+ copy T”ensorFlow/models/research/object_detection/legacy/train.py” script and past it into ‘training-demo’
+initiate training:
$ python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config

III. DETECTION

$ python detect.py










