
/*
  Description: An OpenCV based webcam gaze tracker based on a simple image gradient-based eye center algorithm 
  Platform: VS2015+Opencv3.4
  Date:1/16/2019
  Author:
*/
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

//Necessary Headfiles
#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

// Kernel Function Headers 
void detectAndDisplay( cv::Mat frame );


// Here we  use the detection classifer which was pretrained by Opencv 
cv::String face_cascade_name = "../../Opencv/data/res/haarcascade_frontalface_alt2.xml";

cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection"; // face capture window 
std::string face_window_name = "Capture - Face"; //pupil detection window
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
cv::Mat frame;
/*
  Main function start 
*/
int main( int argc, const char** argv ) {
 // cv::Mat frame;

  // Load the pretrain calssifer 
  printf("\t\t瞳孔识别程序运行中\n\t按下c结束识别，按下s保存图片\n");
  face_cascade.load(face_cascade_name);
  cv::namedWindow(main_window_name,CV_WINDOW_NORMAL); //Create main window 
  cv::moveWindow(main_window_name, 900, 100); // Move Main Window to (400,100)
  cv::namedWindow(face_window_name,CV_WINDOW_NORMAL); //Create face window
  cv::moveWindow(face_window_name, 500, 100);//Move Window to (10,100)
  ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2), 43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

  cv::VideoCapture capture(0);

  if(capture.isOpened()) // The capture is Open 
  {
    while( true ) 
	{
		  capture.read(frame); // read fames from capture
		  
		  cv::flip(frame, frame,1);// mirror it. looks like common 
		  frame.copyTo(debugImage);

		  // Apply the classifier to the frame
		  if( !frame.empty() ) 
		  {
			detectAndDisplay(frame); //Kernel Function 
		  }
		  else 
		  {
			printf(" --(!!!)注意：摄像头检测出错(!!!)-- ");
			break;
		  }
		  imshow(main_window_name,debugImage);  //show the main window, capture your face in this window.RGB

		  // Wait The input key 
		  int c = cv::waitKey(10);
		  if( (char)c == 'c'|| (char)c == 'C' ) // 'c' quit the detection 
		  { break; }
		  if( (char)c == 's'|| (char)c == 'S')
		  {
			imwrite("../../OutputImages/frame.png",frame);
		  }
    }
  }
  return 0;
}

//--------------Operator Functions -------------//
#if 1
/*
*  Find eyes from your face 
*/
void findEyes(cv::Mat frame, cv::Rect face) {
	cv::Mat frame_gray, faceROI;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	cv::Mat faceROI_rgb = frame(face);
	cvtColor(faceROI_rgb, faceROI, CV_BGR2GRAY);
	cv::Mat debugFace = faceROI_rgb;


  //-- Find eye regions and draw them
  int eye_region_width = face.width * (kEyePercentWidth/100.0);
  int eye_region_height = face.width * (kEyePercentHeight/100.0);
  int eye_region_top = face.height * (kEyePercentTop/100.0);
  cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),eye_region_top,eye_region_width,eye_region_height); // left eye region
  cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0), eye_region_top,eye_region_width,eye_region_height); // right eye regin 
  rectangle(debugFace, leftEyeRegion, CV_RGB(0, 255, 0));//green rectangle
  rectangle(debugFace, rightEyeRegion, CV_RGB(0, 255, 0));//green rectangle
  //-- Here !! Find Eye Centers means Find your pupil  
  cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion); 
  cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion); 
  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;
  // draw eye centers
  circle(debugFace, rightPupil, 3, CV_RGB(0, 255, 0));
  circle(debugFace, leftPupil, 3, CV_RGB(0, 255, 0));
  imshow(face_window_name, faceROI_rgb);
}

/*
 *  detectAndDisplay
 */

void detectAndDisplay( cv::Mat frame ) {
  std::vector<cv::Rect> faces;
  //cv::Mat frame_gray;

  std::vector<cv::Mat> rgbChannels(3); // 3 channel Vector 
  cv::split(frame, rgbChannels); // split Mat to a vector
  cv::Mat frame_gray = rgbChannels[2];// get channel 2 signals 
  //cvtColor( frame, frame_gray, CV_BGR2GRAY ); // convert RGB to GRAY 
  //equalizeHist( frame_gray, frame_gray ); //Image quanlity improve if you want 

  //-- Detect faces
  face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150)); //Find your face 
  
  for( int i = 0; i < faces.size(); i++ )
  {
    rectangle(debugImage, faces[i], CV_RGB(0,255,0),3, 8); // generate a green rectangle 
  }
  //-- Show what you got
  // we use gray images  in order to find the pupil quickly 
  if (faces.size() > 0) 
  {
    findEyes(frame, faces[0]); // Find your eyes from your face
  }
}
#endif
