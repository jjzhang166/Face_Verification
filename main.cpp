#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <opencv/cv.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/image_transforms.h>
#include "extract_features.h"
#include "FaceProcessing.h"
#include "JointBayesian.h"
//#include "OpenNiWrapper.h"
//Linux
#include <time.h>

//#define USE_HISTO_EQUAL

int main(int argc, const char** argv)
{  
	//FeatureSelectionUseBoost();
	//return 0;

	// show menu
	printf("--------------------------------------\n");
	printf(" Face Verification(VGG) (video)\n");
	printf("--------------------------------------\n");

	// use camera, ASUS xtion, or video as input source
	//cv::VideoCapture cap(0);
	//COpenNiWrapper openNiWrapper;
	cv::gpu::setDevice(0);
	printf("Setting GPU...\n");

	/*if (argc != 5) {
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< " mean.binaryproto labels.txt " << std::endl;
		return 1;
	}*/
	::google::InitGoogleLogging(argv[0]);
	string model_file   = "model/VGG_FACE_deploy.prototxt";
	string trained_file = "model/VGG_FACE.caffemodel";
	string mean_file    = "model/mean.binaryproto";
	string label_file   = "model/names.txt";
	Classifier classifier(model_file, trained_file, mean_file, label_file);
	//cv::VideoCapture cap("/home/sylar/gender_classification/Backstreet.mp4");
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) return -1;

	// load the cascades
	/*CFaceProcessing fp("D:/Software/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml",
	  "D:/Software/opencv/sources/data/haarcascades/haarcascade_mcs_nose.xml",
	  "D:/Software/opencv/sources/data/haarcascades/haarcascade_mcs_mouth.xml",
	  "D:/Vision_Project/shape_predictor_68_face_landmarks.dat");*/

	//[20160104_Sylar]
	CFaceProcessing fp("/home/sylar/gender_classification/xml/lbpcascade_frontalface.xml",
			"/home/sylar/gender_classification/xml/haarcascade_mcs_nose.xml",
			"/home/sylar/gender_classification/xml/haarcascade_mcs_mouth.xml",
			"/home/sylar/gender_classification/xml/shape_predictor_68_face_landmarks.dat");

	// Joint Bayesian
	Py_Initialize();
	const int SIZE = 4096;
	const int threshold = 50;
	CJointBayesian jointbayesian(SIZE, threshold);
	
	
	// main loop
	bool testperson = false;
	float* testp = new float[SIZE];
	cv::Mat img;
	bool showLandmark = false;
	bool showCroppedFaceImg = false;	
	cv::Mat grayFrame;	
	cv::Mat grayFramePrev;
	std::vector<std::vector<cv::Point> > fLandmarksPrev;
	std::vector<std::vector<cv::Point> > fLandmarks;
	std::vector<unsigned char> faceStatusPrev;
	std::vector<float> accGenderConfidencePrev;
	float totalCount = 0;
	float falseCount = 0;
	std::vector<cv::Mat> prevCropped;
	bool enable_gpu = false;
	cv::TickMeter timer;
	timer.start();
	while (1)
	{  
		//openNiWrapper.GetDepthColorRaw();
		//openNiWrapper.ConvertDepthColorRawToImage(cv::Mat(), img);
		cap >> img;
		if (img.empty()) break;
		cv::resize(img, img, cv::Size(1280, 720));
		// (optional) backup original image for offline debug
		cv::Mat originImg(img.size(), img.type());
		img.copyTo(originImg);

		// time calculation
		cv::TickMeter tm;
		tm.start();

		// -----------------------------------
		// face detection 
		// -----------------------------------
		std::vector<cv::Rect> faces;
		// -----------------
		// consume most time
		int faceNum ;
		// Press g to change mode
		if(enable_gpu){
			faceNum = fp.FaceDetection_GPU(img);
			cv::putText(img, "GPU", cv::Point(30, 100), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 255, 255));  
		}
		else{
			faceNum = fp.FaceDetection(img);
			cv::putText(img, "CPU", cv::Point(30, 100), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 255, 255)); 
		}
		// -----------------
		std::vector<cv::Mat> croppedImgs;
		if (faceNum > 0)
		{
			faces = fp.GetFaces();

			// normalize the face image with landmark
			std::vector<cv::Mat> normalizedImg;
			fp.AlignFaces2D(normalizedImg, originImg);
			// ----------------------------------------
			// crop faces and do histogram equalization
			// ----------------------------------------
			croppedImgs.resize(faceNum);
			for (int i = 0; i < faceNum; i++)
			{
				// ------------------------------
				// Sylar 20160308 to use RGBscale
				// ------------------------------
				int x = faces[i].x - (faces[i].width / 4);
				int y = faces[i].y - (faces[i].height / 4);
				if (x < 0)
					x = 0;
				if (y < 0)
					y = 0;
				int w = faces[i].width + (faces[i].width / 2) ;
				int h = faces[i].height + (faces[i].height / 2);
				if(w + x > originImg.cols)
					w = originImg.cols - x ;
				if(h + y > originImg.rows)
					h = originImg.rows - y ;
				croppedImgs[i] = originImg(cv::Rect(x, y, w, h)).clone();

		// --------------------------------------------
		// do gender classification and display results
		// --------------------------------------------
		std::vector<unsigned char> status = fp.GetFaceStatus();
		for (int i = 0; i < faceNum; i++)
		{
			if (status[i])
			{   
                // Test person
				if (!testperson){          
					cv::imshow("rrr", croppedImgs[i]);
				    char key = (char)cv::waitKey(); 
				    if (key==32){    
				        const float* temp = classifier.Classify(croppedImgs[i]);  
					    for (int d=0; d<SIZE; d++){
					        testp[d] = temp[d];
					    }
					    testperson = true;
				    }    
				}
				else if(testperson){
					// Face Verify
	                const float* features;   
					features = classifier.Classify(croppedImgs[i]);
					bool result = jointbayesian.Verify(const_cast<float*>(features), testp);                   
					result ? printf("Same\n") : printf("Different\n");
					std::cout<<accum<<std::endl;
					cv::imshow("Verify", croppedImgs[i]);
					cv::waitKey(1);
				}
			}

		}
		// show processing time
		//clock_t eTime = clock();
		tm.stop();
		double detectionTime = tm.getTimeMilli();
		double fps = 1000 / detectionTime;
		char deltaTimeStr[256] = { 0 };
		//sprintf(deltaTimeStr, "%d ms", (double)(eTime - sTime));.
		sprintf(deltaTimeStr, "%f fps", fps);
		cv::putText(img, deltaTimeStr, cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 255, 255));      
		cv::imshow("Result", img);
		//if (faceNum > 0) key = cv::waitKey();
		//else key = cv::waitKey(1);
		char key = (char)cv::waitKey(10);

		if (key == 27) break;
		else if (key == 83 || key == 115)
		{
			std::time_t time = std::time(NULL);
			char timeStr[128] = { 0 };
			std::strftime(timeStr, sizeof(timeStr), "./Offline/%Y-%m-%d-%H-%M-%S.bmp", std::localtime(&time));
			cv::imwrite(timeStr, originImg);
		}
		else if (key == 76 || key == 108) // 'l' or 'L'
		{
			showLandmark = !showLandmark;
		}
		else if (key == 70 || key == 102) // 'f' or 'F'
		{
			showCroppedFaceImg = !showCroppedFaceImg;
		}
		else if (key == 71 || key == 103)
		{
			enable_gpu = !enable_gpu;
		}
		fp.CleanFaces();
	}
	timer.stop();
	double wholeTime = timer.getTimeSec();
	std::cout<< "time "<<wholeTime<<std::endl;
	std::cout << "False Rate :"<<falseCount <<"/"<< totalCount << std::endl;

	Py_Finalize();
	//
	return 0;
}


