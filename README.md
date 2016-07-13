# Face_Verification
Using deep learning model as feature extractor and joint-bayesian as classifier, to classify two faces belong to same person or different people.
# Requirements
* OpenCV
* Caffe
* dlib
* caffe trained model in model/
* joint bayesian model in result/
* dlib and opencv face detection model in xml/

# Usage
```
cmake .
make
```
In main.cpp:

```C++
const int SIZE = 4096; // feature vector's size from target layer
const int threshold = 50; // ratio threshold from joint bayesian
const char* layer_name = "fc7"; // target layer name
```
At the beginning, press enter to see the detected face and press space to choose the shown image to be ground truth.




