# Traffic_sign_detectionNrecognition

This Project is based on the tutorial: 
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

Here we have an overview of steps I followed:

(PS: I used Anaconda python distribution on ubuntu 16 .04)<br>
I. Installation of requirements:<br>
1. Tensorflow API<br>
+ Create new conda environment:<br>
conda create -n tensorflow_cpu pip python=3.6<br>
activate tensorflow_cpu<br>

+ install tensorflow:<br>
pip install --ignore-installed --upgrade tensorflow==1.9<br>

2. TensorFlow Models Installation<br>
+ install prerequisits<br>
$ conda install pillow, xml, jupyter, opencv, cython<br>
the versions packages above used are:<br>

+ create new folder with the name TensorFlow<br>
(ex: /home/<user>/TensorFlow)<br>
+ cd into TensorFlow directory<br>
+ clone the model in side TensorFlow directory<br>
$ git clone https://github.com/tensorflow/models.git<br>

3. Protobuf installation/compilation:<br>
+ Install protobuf using Homebrew:<br>
$ brew install protobuf<br>
+ adding environment variables:<br>
$ cd <path_to_your_TensorFlow_directory>/models/research/<br>
$ protoc object_detection/protos/*.proto –python_out=.<br>
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim<br>

(!!! The commands above should be run whenever we open new terminal)<br>
From <path_to_your_TensorFlow_directory>/models/research/ , run:<br>
$ python setup.py build<br>
$ python setup.py install<br>
(!!! we run the two commands above whenever we change or update the object-detection python package)<br>

4. LabelImg Installation<br>
+ create new conda environment<br>
$ conda create -n labelImg pyqt=5<br>
$ activate labelImg<br>
+ inside TensorFlow directory, we create new directory ‘addons’<br>

TensorFlow<br>
... addons<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... labelImg<br>
... models<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... official<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... research<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... samples<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... tutorials<br>
    
+ clone labelImg repo<br>
$ git clone https://github.com/tzutalin/labelImg.git<br>
+ install dependencies:<br>
$ sudo apt-get install pyqt5-dev-tools<br>
$ sudo pip install -r requirements/requirements-linux-python3.txt<br>
$ make qt5py3<br>
+ test<br>
$ cd TensorFlow\addons\labelImg<br>
$ python labelImg.py<br>

II. Training Custom Object Detection:<br>
1. Workspace organization<br>
+ create new folder “workspace” under TensorFlow directory, and another one “training-demo” under “workspace” just have been created:<br>

... addons<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... labelImg<br>
... models<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... official<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... research<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... samples<br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... tutorials<br>
... workspace<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... training_demo<br>
    
+ under “training-demo”  folder we add sub-folders: <br>

... annotations<br>
... images<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... test<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... train<br>
... pre-trained-model<br>
... training<br>
... README.md<br>

2. preparing annotations<br>
here we use labelImg to annotate our own dataset. One we collect images for diffrent classes we start labeling operation
(I used 3 classes each one contains 100 samples, 90% for training and 10% for testing)<br>
$ activate labelImg<br>
$ python labelImg.py ..\..\workspace\training_demo\images<br>
+ creating Label Map<br>
a file that maps each label (class name) to an id [label_map.pbtxt]. It should be placed in<br>
training-demo/annotations folder.<br>
3. generate tf records<br>
we add “scripts” and “scripts/preprocessing” folders<br>

TensorFlow<br>
... addons<br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... labelImg<br>
... models<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... official<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... research<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... samples<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... tutorials<br>
... scripts<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... preprocessing<br>
... works<br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... training_demo<br>

+ converting *.xml to *.csv<br>
- under ‘scripts/preprocessing’run xml_to_csv.py script <br>  
$ python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv<br>

$ python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv<br>

→ 2 files will be generated under ‘training-demo/annotations’ : test_label.csv and train_label.csv<br>

+ Converting from *.csv to *.record<br>
- under ‘scripts/preprocessing’generate_tfrecord.py script<br>
$ python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv<br>
--img_path=<PATH_TO_IMAGES_FOLDER>/train  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record<br>

$ python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv<br>
--img_path=<PATH_TO_IMAGES_FOLDER>/test<br>
--output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record<br>

→ 2 files will be generated under ‘training-demo/annotations’ : test.record and train.record<br>
4. configure training<br>
+ download configuration file of the model choosed (in this case: ssd_mobilenet_v3_coco.config) from:
https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs<br>

+ we need also to download the pre-trained NN<br>
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models<br>

+ we extract the content of *.tar.gz inside ‘training-demo/pre-trained-model’<br>

+ follow comment of .config file and make the modifications<br>
(.config file must be saved in ‘training-demo/training directory’)<br>

5. train the model<br>
+ copy T”ensorFlow/models/research/object_detection/legacy/train.py” script and past it into ‘training-demo’<br>
+initiate training:<br>
$ python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config<br>

III. DETECTION<br>

$ python detect.py










