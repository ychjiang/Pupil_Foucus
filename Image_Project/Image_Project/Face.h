#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class Line
{
public:
	//��㡢�յ㡢�е㡢�߳����Ƕ�
	Point2f P = Point2f(0.f, 0.f);
	Point2f Q = Point2f(0.f, 0.f);
	Point2f M = Point2f(0.f, 0.f);
	double len{ 0 };
	float degree{ 0 };

	//����a,b,p
	double a = 1.0;
	double b = 2.0;
	double p = 1.0;

public:
	Line() :P(Point2f(0.f, 0.f)), Q(Point2f(0.f, 0.f)) {};

	//��P,Q����õ��е㣬�߳����Ƕ���Ϣ 
	void PQtoMLD();

	void MLDtoPQ();
	void show();

	double Getu(Point2f X);
	double Getv(Point2f X);
	Point2f Get_Point(double u, double v);
	double Get_Weight(Point2f X);
};


class LinePair
{
public:
	Line leftLine;
	Line rightLine;
	vector<Line> warpLine;

public:

	//�����м���ɵ��߶μ���
	void genWarpLine(int frame_count);
};


class Image
{
public:

	//����֡���
	int frame_index;

	Mat left_image;
	Mat right_image;
	Mat new_image;

public:
	Image(int frame_index, string leftImageName, string rightImageName);

	//˫���Բ�ֵ
	Vec3b bilinear(Mat image, double X, double Y);

	//���ɹ���ͼ��֡
	void Warp(int frame_count, string new_image_name, vector<LinePair> pairs);
};


class Morpher
{
public:
	vector<LinePair> pairs;
	LinePair curLinePair;

	//����������ʽ����Ҫ�õ�
	int counter = 0;

	//���ö����Ĺ���֡������1Ϊ50%��3Ϊ25%��50%��75%
	int frame_count = 1;

	Mat leftImage;
	Mat rightImage;
	Mat leftImageTmp;
	Mat rightImageTmp;

	//��ʾ��ء����ߵ���ɫ���߿�ƫ��
	Scalar color = Scalar(0, 255, 0);
	int thickness = 2;
	int shift = 0;

	//����ֵ�����ڿ���
	int key;

	string first_image_name;
	string second_image_name;
	string new_image_name;


public:
	void show_pairs();

	//������ͼ���ϵ���궯��
	static void on_mousel(int event, int x, int y, int flag, void* param);

	//������ͼ���ϵ���궯��
	static void on_mouser(int event, int x, int y, int flag, void* param);

	//����morph
	void runWarp();

	void main();

};


#if 1
#include"Morpher.h"
#include "images_headfile.h"
void Line::PQtoMLD()
{
	M.x = (P.x + Q.x) / 2;
	M.y = (P.y + Q.y) / 2;

	float tmpx = Q.x - P.x;
	float tmpy = Q.y - P.y;

	len = sqrt(tmpx*tmpx + tmpy*tmpy);
	degree = atan2(tmpy, tmpx);
	return;
}

void Line::MLDtoPQ()
{
	float tmpx = 0.5*len*cos(degree);
	float tmpy = 0.5*len*sin(degree);

	Point2f tmpP;
	Point2f tmpQ;
	tmpP.x = M.x - tmpx;
	tmpP.y = M.y - tmpy;
	tmpQ.x = M.x + tmpx;
	tmpQ.y = M.y + tmpy;

	P = tmpP;
	Q = tmpQ;
	return;
}

void Line::show()
{
	printf_s("P(%lf,%lf) Q(%lf,%lf) M(%lf,%lf)\n \tlen=%lf degree=%f\n", P.x, P.y, Q.x, Q.y, M.x, M.y, len, degree);
	return;
}

double Line::Getu(Point2f X)
{
	double X_P_x = X.x - P.x;
	double X_P_y = X.y - P.y;
	double Q_P_x = Q.x - P.x;
	double Q_P_y = Q.y - P.y;
	double u = ((X_P_x * Q_P_x) + (X_P_y * Q_P_y)) / (len * len);
	return u;
}

double Line::Getv(Point2f X)
{
	double X_P_x = X.x - P.x;
	double X_P_y = X.y - P.y;
	double Q_P_x = Q.x - P.x;
	double Q_P_y = Q.y - P.y;
	double Perp_Q_P_x = Q_P_y;
	double Perp_Q_P_y = -Q_P_x;
	double v = ((X_P_x * Perp_Q_P_x) + (X_P_y * Perp_Q_P_y)) / len;
	return v;
}

Point2f Line::Get_Point(double u, double v)
{
	double Q_P_x = Q.x - P.x;
	double Q_P_y = Q.y - P.y;
	double Perp_Q_P_x = Q_P_y;
	double Perp_Q_P_y = -Q_P_x;
	double Point_x = P.x + u * (Q.x - P.x) + ((v * Perp_Q_P_x) / len);
	double Point_y = P.y + u * (Q.y - P.y) + ((v * Perp_Q_P_y) / len);
	Point2f X;
	X.x = Point_x;
	X.y = Point_y;
	return X;
}

double Line::Get_Weight(Point2f X)
{
	double d = 0.0;

	double u = Getu(X);
	if (u > 1.0)
		d = sqrt((X.x - Q.x) * (X.x - Q.x) + (X.y - Q.y) * (X.y - Q.y));
	else if (u < 0)
		d = sqrt((X.x - P.x) * (X.x - P.x) + (X.y - P.y) * (X.y - P.y));
	else
		d = abs(Getv(X));


	double weight = pow(pow(len, p) / (a + d), b);
	return weight;
}




void LinePair::genWarpLine(int frame_count)
{
	while (leftLine.degree - rightLine.degree>3.14159265)
		rightLine.degree = rightLine.degree + 3.14159265;

	while (rightLine.degree - leftLine.degree>3.14159265)
		leftLine.degree = leftLine.degree + 3.14159265;

	for (int i = 0; i<frame_count; i++)
	{
		double ratio = (double)(i + 1) / (frame_count + 1);
		Line curLine;

		curLine.M.x = (1 - ratio)*leftLine.M.x + ratio*rightLine.M.x;
		curLine.M.y = (1 - ratio)*leftLine.M.y + ratio*rightLine.M.y;
		curLine.len = (1 - ratio)*leftLine.len + ratio*rightLine.len;
		curLine.degree = (1 - ratio)*leftLine.degree + ratio*rightLine.degree;

		curLine.MLDtoPQ();
		warpLine.push_back(curLine);
	}
	return;
}

Image::Image(int index, string leftImageName, string rightImageName)
{
	frame_index = index;

	left_image = imread(leftImageName);
	right_image = imread(rightImageName);

	Size ImageSize = Size(left_image.cols, left_image.rows);
	new_image.create(ImageSize, CV_8UC3);
}

Vec3b Image::bilinear(Mat image, double X, double Y)
{
	int width = image.cols;
	int height = image.rows;

	int x_floor = (int)X;
	int y_floor = (int)Y;
	int x_ceil = x_floor + 1;
	int y_ceil = y_floor + 1;
	double a = X - x_floor;
	double b = Y - y_floor;
	if (x_ceil >= width - 1)
		x_ceil = width - 1;
	if (y_ceil >= height - 1)
		y_ceil = height - 1;
	Vec3b output_scalar;
	Vec3b leftdown = image.at<Vec3b>(y_floor, x_floor);
	Vec3b lefttop = image.at<Vec3b>(y_ceil, x_floor);
	Vec3b rightdown = image.at<Vec3b>(y_floor, x_ceil);
	Vec3b righttop = image.at<Vec3b>(y_ceil, x_ceil);

	for (int i = 0; i < 3; i++)
	{
		output_scalar.val[i] = (1 - a)*(1 - b)*leftdown.val[i] + a*(1 - b)*rightdown.val[i] + a*b*righttop.val[i] + (1 - a)*b*lefttop.val[i];
	}
	return output_scalar;
}

void Image::Warp(int frame_count, string new_image_name, vector<LinePair> pairs)
{
	double ratio = (double)(frame_index + 1) / (frame_count + 1);

	Mat ori_leftImage, ori_rightImage;
	ori_leftImage = left_image;
	ori_rightImage = right_image;

	int width = new_image.cols;
	int height = new_image.rows;


	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			Point2f dst_point;
			dst_point.x = x;
			dst_point.y = y;
			double leftXSum_x = 0.0;
			double leftXSum_y = 0.0;
			double leftWeightSum = 0.0;
			double rightXSum_x = 0.0;
			double rightXSum_y = 0.0;
			double rightWeightSum = 0.0;
			for (int i = 0; i < pairs.size(); i++) {
				Line src_line = pairs[i].leftLine;
				Line dst_line = pairs[i].warpLine[frame_index];

				double new_u = dst_line.Getu(dst_point);
				double new_v = dst_line.Getv(dst_point);

				Point2f src_point = src_line.Get_Point(new_u, new_v);
				double src_weight = dst_line.Get_Weight(dst_point);
				leftXSum_x = leftXSum_x + (double)src_point.x * src_weight;
				leftXSum_y = leftXSum_y + (double)src_point.y * src_weight;
				leftWeightSum = leftWeightSum + src_weight;

				src_line = pairs[i].rightLine;

				new_u = dst_line.Getu(dst_point);
				new_v = dst_line.Getv(dst_point);

				src_point = src_line.Get_Point(new_u, new_v);
				src_weight = dst_line.Get_Weight(dst_point);
				rightXSum_x = rightXSum_x + (double)src_point.x * src_weight;
				rightXSum_y = rightXSum_y + (double)src_point.y * src_weight;
				rightWeightSum = rightWeightSum + src_weight;
			}

			double left_src_x = leftXSum_x / leftWeightSum;
			double left_src_y = leftXSum_y / leftWeightSum;
			double right_src_x = rightXSum_x / rightWeightSum;
			double right_src_y = rightXSum_y / rightWeightSum;

			if (left_src_x<0)
				left_src_x = 0;
			if (left_src_y<0)
				left_src_y = 0;
			if (left_src_x >= width)
				left_src_x = width - 1;
			if (left_src_y >= height)
				left_src_y = height - 1;
			if (right_src_x<0)
				right_src_x = 0;
			if (right_src_y<0)
				right_src_y = 0;
			if (right_src_x >= width)
				right_src_x = width - 1;
			if (right_src_y >= height)
				right_src_y = height - 1;


			Vec3b left_scalar = bilinear(ori_leftImage, left_src_x, left_src_y);
			Vec3b right_scalar = bilinear(ori_rightImage, right_src_x, right_src_y);
			Vec3b new_scalar;
			new_scalar.val[0] = (1 - ratio)*left_scalar.val[0] + ratio*right_scalar.val[0];
			new_scalar.val[1] = (1 - ratio)*left_scalar.val[1] + ratio*right_scalar.val[1];
			new_scalar.val[2] = (1 - ratio)*left_scalar.val[2] + ratio*right_scalar.val[2];
			new_image.at<Vec3b>(y, x) = new_scalar;
		}

	}
	char win_name[16];
	char img_name[50];
	printf_s(img_name, "%s_%d.jpg", new_image_name.c_str(), frame_index);

	imshow(img_name, new_image);
	imwrite("HH.jpg", new_image);
	return;
}

void Morpher::runWarp()
{
	for (int i = 0; i<frame_count; i++)
	{
		Image curImage = Image(i, first_image_name, second_image_name);
		printf_s("warping %d...\n", i);
		curImage.Warp(frame_count, new_image_name, pairs);
	}
}



void Morpher::show_pairs()
{
	int len = pairs.size();
	printf("pairs size=%d\n", len);
	for (int i = 0; i<len; i++)
	{
		printf("leftLine:");
		pairs[i].leftLine.show();
		printf("rightLine:");
		pairs[i].rightLine.show();
		printf("\n");
	}
}

void Morpher::on_mousel(int event, int x, int y, int flag, void* param)
{
	Morpher* pm = (Morpher*)param;
	int& counter = pm->counter;
	LinePair& curLinePair = pm->curLinePair;
	Mat& leftImage = pm->leftImage;
	Mat& leftImageTmp = pm->leftImageTmp;
	Scalar& color = pm->color;
	int& thickness = pm->thickness;
	int& shift = pm->shift;

	if (counter % 2 == 0 && pm->counter > 0)
	{
		curLinePair.warpLine.clear();
		if (event == CV_EVENT_LBUTTONDOWN || event == CV_EVENT_RBUTTONDOWN) {
			printf("Left image( %d, %d) ", x, y);
			printf("The Event is : %d ", event);
			printf("The flags is : %d ", flag);
			printf("The param is : %d\n", param);
			curLinePair.leftLine.P.x = x;
			curLinePair.leftLine.P.y = y;
			cout << "P:" << curLinePair.leftLine.P.x << '\t' << curLinePair.leftLine.P.y << endl;
		}
		if (event == CV_EVENT_LBUTTONUP || event == CV_EVENT_RBUTTONUP) {
			printf("Left image( %d, %d) ", x, y);
			printf("The Event is : %d ", event);
			printf("The flags is : %d ", flag);
			printf("The param is : %d\n", param);
			curLinePair.leftLine.Q.x = x;
			curLinePair.leftLine.Q.y = y;
			curLinePair.leftLine.PQtoMLD();
			cout << "P:" << curLinePair.leftLine.P.x << '\t' << curLinePair.leftLine.P.y << endl;
			cout << "Q:" << curLinePair.leftLine.Q.x << '\t' << curLinePair.leftLine.Q.y << endl;
			line(leftImage, curLinePair.leftLine.P, curLinePair.leftLine.Q, color, thickness, CV_AA, shift);
			leftImageTmp = leftImage.clone();
			counter--;
		}
		if (flag == CV_EVENT_FLAG_LBUTTON || flag == CV_EVENT_FLAG_RBUTTON) {
			curLinePair.leftLine.Q.x = x;
			curLinePair.leftLine.Q.y = y;
			leftImage.release();
			leftImage = leftImageTmp.clone();
			line(leftImage, curLinePair.leftLine.P, curLinePair.leftLine.Q, color, thickness, CV_AA, shift);
			imshow("left", leftImage);
		}
	}
	else
	{
		//cout << "you have selected the wrong image" << endl;
	}
}

void Morpher::on_mouser(int event, int x, int y, int flag, void* param)
{
	Morpher* pm = (Morpher*)param;
	int& counter = pm->counter;
	LinePair& curLinePair = pm->curLinePair;
	Mat& rightImage = pm->rightImage;
	Mat& rightImageTmp = pm->rightImageTmp;
	int& frame_count = pm->frame_count;
	vector<LinePair>& pairs = pm->pairs;
	Scalar& color = pm->color;
	int& thickness = pm->thickness;
	int& shift = pm->shift;

	if (counter % 2 == 1 && counter > 0)
	{
		if (event == CV_EVENT_LBUTTONDOWN || event == CV_EVENT_RBUTTONDOWN) {
			printf("right image( %d, %d) ", x, y);
			printf("The Event is : %d ", event);
			printf("The flags is : %d ", flag);
			printf("The param is : %d\n", param);
			curLinePair.rightLine.P.x = x;
			curLinePair.rightLine.P.y = y;
		}
		if (event == CV_EVENT_LBUTTONUP || event == CV_EVENT_RBUTTONUP) {
			printf("right image( %d, %d) ", x, y);
			printf("The Event is : %d ", event);
			printf("The flags is : %d ", flag);
			printf("The param is : %d\n", param);
			curLinePair.rightLine.Q.x = x;
			curLinePair.rightLine.Q.y = y;
			curLinePair.rightLine.PQtoMLD();
			line(rightImage, curLinePair.rightLine.P, curLinePair.rightLine.Q, color, thickness, CV_AA, shift);
			rightImageTmp = rightImage.clone();
			counter--;
			curLinePair.genWarpLine(frame_count);
			pairs.push_back(curLinePair);

			printf("\n");
			pm->show_pairs();

			counter = 0;
		}
		if (flag == CV_EVENT_FLAG_LBUTTON || flag == CV_EVENT_FLAG_RBUTTON) {
			curLinePair.rightLine.Q.x = x;
			curLinePair.rightLine.Q.y = y;
			rightImage.release();
			rightImage = rightImageTmp.clone();
			line(rightImage, curLinePair.rightLine.P, curLinePair.rightLine.Q, color, thickness, CV_AA, shift);
			imshow("right", rightImage);
		}
	}
	else
	{
		//cout << "you have selected the wrong image" << endl;
	}
}

/*
void Morpher::main()
{
	first_image_name = "../../../picture/1.jpg";
	second_image_name = "../../../picture/5.jpg";
	new_image_name = "morph";

	//���ö����Ĺ���֡��������1Ϊ50%��3Ϊ25%��50%��75%
	frame_count = 3;

	leftImage = imread(first_image_name);
	rightImage = imread(second_image_name);

	leftImageTmp = leftImage.clone();
	rightImageTmp = rightImage.clone();

	namedWindow("left", 1);
	moveWindow("left", 10, 10);
	namedWindow("right", 1);
	moveWindow("right", 300, 10);

	setMouseCallback("left", on_mousel, this);
	setMouseCallback("right", on_mouser, this);

	imshow("left", leftImage);
	imshow("right", rightImage);

	while (true)
	{
		key = waitKey(0);
		if (key == 'c')
			counter = counter + 2;
		else if (key == 'w')
			runWarp();
		else if (key == 'q')
			break;
	}

	imwrite("left_marked.jpg", leftImage);
	imwrite("right_marked.jpg", rightImage);

	return;
}

*/
#endif

