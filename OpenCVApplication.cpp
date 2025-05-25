// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

wchar_t* projectPath;

void compute_object_properties_helper(int x, int y, Mat& img);

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
			printf("Computing object's characteristics...\n");
			compute_object_properties_helper(x, y, *src);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void additive_factor(Mat& img, int additive) {


	int height = img.rows;
	int width = img.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			// Step 1: Add a value (positive or negative) to the pixel at (i, j) and store it in an integer
			// Be careful when adding or subtracting from an unsigned char (0-255 range).
			int temp_value = img.at<uchar>(i, j) + additive;

			// Step 2: Handle potential overflow or underflow: : Clamp the result
			// - Overflow: If the result exceeds 255, store 255 (maximum value).
			// - Underflow: If the result is less than 0, store 0 (minimum value).
			if (temp_value > 255)
				dst.at<uchar>(i, j) = 255;
			else
			{
				if (temp_value < 0)
					dst.at<uchar>(i, j) = 0;
				else
					dst.at<uchar>(i, j) = temp_value;
			}
		}
	}
	imshow("additive factor", dst);
	waitKey(0);

}

void multiplicative_factor(Mat& img, int multiplicative)
{
	int height = img.rows;
	int width = img.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			// Step 1: Multiply with a float value the pixel at (i, j) and store result in a integer
			
			int temp_value = img.at<uchar>(i, j) * multiplicative;
			// Step 2: Handle potential overflow: Clamp the result
			// - Overflow: If the result exceeds 255, store 255 (maximum value)
			if (temp_value > 255)
				dst.at<uchar>(i, j) = 255;
			else
			{
				if (temp_value < 0)
					dst.at<uchar>(i, j) = 0;
				else
					dst.at<uchar>(i, j) = temp_value;
			}
		}
	}

	imshow("multiplicative factor", dst);
	waitKey(0);
}

void color_squares() {

	Mat img(256, 256, CV_8UC3);

	for (int i = 0; i < img.rows / 2; i++) {
		for (int j = 0; j < img.cols / 2; j++) {
			img.at<Vec3b>(i, j)[0] = 255;
			img.at<Vec3b>(i, j)[1] = 255;
			img.at<Vec3b>(i, j)[2] = 255;
		}
	}

	for (int i = img.rows / 2; i < img.rows; i++) {
		for (int j = 0; j < img.cols / 2; j++) {
			img.at<Vec3b>(i, j)[0] = 0;
			img.at<Vec3b>(i, j)[1] = 255;
			img.at<Vec3b>(i, j)[2] = 0;
		}
	}

	for (int i = 0; i < img.rows / 2; i++) {
		for (int j = img.cols / 2; j < img.cols; j++) {
			img.at<Vec3b>(i, j)[0] = 0;
			img.at<Vec3b>(i, j)[1] = 0;
			img.at<Vec3b>(i, j)[2] = 255;
		}
	}

	for (int i = img.rows / 2; i < img.rows; i++) {
		for (int j = img.cols / 2; j < img.cols; j++) {
			img.at<Vec3b>(i, j)[0] = 0;
			img.at<Vec3b>(i, j)[1] = 255;
			img.at<Vec3b>(i, j)[2] = 255;
		}
	}

	imshow("multiplicative factor", img);
	waitKey(0);
}


void color_inverse() {
	float vals[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9};
	Mat M(3, 3, CV_64F, vals);
	std::cout << M << std::endl << std::endl;

	Mat M_inv = M.inv();
	std::cout << M_inv << std::endl;
	system("pause");
}

void driver_function_l1() {

	Mat img = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/Lab1/L1/PI-Images/cameraman.bmp", IMREAD_GRAYSCALE);
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Excersize 3 additive factor\n");
		printf(" 2 - Excersize 4 multiplicative factor\n");
		printf(" 3 - Excersize 5 color squares\n");
		printf(" 4 - Excersize 6 color inverse\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			additive_factor(img, -150);
			break;
		case 2:
			multiplicative_factor(img, 2);
			break;
		case 3:
			color_squares();
			break;
		case 4:
			color_inverse();
			break;	
		}

	} while (op != 0);
}

void l2_copy_rgb(Mat& img) {

	int height = img.rows;
	int width = img.cols;

	Mat dst_r = Mat(height, width, CV_8UC1);
	Mat dst_g = Mat(height, width, CV_8UC1);
	Mat dst_b = Mat(height, width, CV_8UC1);


	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			dst_r.at<uchar>(i, j)= img.at<Vec3b>(i, j)[2];

			dst_g.at<uchar>(i, j) = img.at<Vec3b>(i, j)[1];

			dst_b.at<uchar>(i, j) = img.at<Vec3b>(i, j)[0];
		}
	}

	imshow("dst_r", dst_r);
	imshow("dst_g", dst_g);
	imshow("dst_b", dst_b);
	waitKey(0);
}

void l2_convert_color_to_grayscale(Mat& img) {

	int height = img.rows;
	int width = img.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			dst.at<uchar>(i, j) = (img.at<Vec3b>(i, j)[0] + img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j)[2])/3;
		}
	}

	imshow("dst", dst);
	waitKey(0);
}

void l2_convert_grayscale_to_binary(Mat& img) {

	int height = img.rows;
	int width = img.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			if (img.at<uchar>(i, j) > 128)
				dst.at<uchar>(i, j) = 255;
			else
				dst.at<uchar>(i, j) = 0;
		}
	}

	imshow("dst", dst);
	waitKey(0);
}


float max_three(float a, float b, float c) {
	float m1 = max(a, b);
	if (c > m1)
		return c;
	else
		return m1;
}

float min_three(float a, float b, float c) {
	float m1 = min(a, b);
	if (c < m1)
		return c;
	else
		return m1;
}

void hsv_conversion(Mat& img) {

	Size img_size = Size(img.cols, img.rows);
	int height = img.rows;
	int width = img.cols;
	Mat dst1 = Mat(height, width, CV_8UC1);
	Mat dst2 = Mat(height, width, CV_8UC1);
	Mat dst3 = Mat(height, width, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			// Step 1: Multiply with a float value the pixel at (i, j) and
			//store result in a integer
				// Step 2: Handle potential overflow: Clamp the result
				// - Overflow: If the result exceeds 255, store 255 (maximum
				//value)
			float r, g, b;
			r = img.at<Vec3b>(i, j)[2] / 255.0;
			g = img.at<Vec3b>(i, j)[1] / 255.0;
			b = img.at<Vec3b>(i, j)[0] / 255.0;

			float M = max_three(r, g, b);
			float m = min_three(r, g, b);
			float C = M - m;

			float V = M;
			float S, H;
			if (V != 0)
				S = C / V;
			else
				S = 0;

			if (C != 0) {
				if (M == r)
					H = 60 * (g - b) / C;
				if (M == g)
					H = 120 + 60 * (b - r) / C;
				if (M == b)
					H = 240 + 60 * (r - g) / C;

			}
			else
			{
				H = 0;
			}

			if (H < 0)
				H = H + 360;

			uchar H_norm = (H * 255) / 360;
			uchar S_norm = S * 255;
			uchar V_norm = V * 255;

			dst1.at<uchar>(i, j) = H_norm;
			dst2.at<uchar>(i, j) = S_norm;
			dst3.at<uchar>(i, j) = V_norm;


		}
	}


	const char* WIN_DST1 = "Dst_1"; //window for the destination (processed) image
	namedWindow(WIN_DST1, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST1, img_size.width + 30, 0);
	imshow(WIN_DST1, dst1);

	const char* WIN_DST2 = "Dst_2"; //window for the destination (processed) image
	namedWindow(WIN_DST2, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST2, img_size.width + 30, 0);
	imshow(WIN_DST2, dst2);

	const char* WIN_DST3 = "Dst_3"; //window for the destination (processed) image
	namedWindow(WIN_DST3, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST3, img_size.width + 30, 0);
	imshow(WIN_DST3, dst3);

	waitKey(0);

}



void driver_function_l2() {

	Mat img = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/Lab1/L1/PI-Images/Lena_24bits.bmp", IMREAD_COLOR);
	Mat img_gray = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/Lab1/L1/PI-Images/cameraman.bmp", IMREAD_GRAYSCALE);
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Excersize 1 copy R, G, B in 3 different windows\n");
		printf(" 2 - Excersize 2 convert color to grayscale\n");
		printf(" 3 - Excersize 3 convert grayscale to binary\n");
		printf(" 4 - Excersize 4 H, S, V from R, G, B\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			l2_copy_rgb(img);
			break;
		case 2:
			l2_convert_color_to_grayscale(img);
			break;
		case 3:
			l2_convert_grayscale_to_binary(img_gray);
			break;
		case 4:
			hsv_conversion(img);
			break;
		}


	} while (op != 0);
}


void compute_histogram(Mat& img) {
	int height = img.rows;
	int width = img.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	//histogram
	std::vector<int> hist(256, 0);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			//hist.push_back(img.at<uchar>(i, j));
			hist[img.at<uchar>(i, j)]++;
		}
	}

	showHistogram("histogram image", &hist[0], 256, 256);
	waitKey(0);
}

void compute_pdf(Mat& img) {
	int height = img.rows;
	int width = img.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	//histogram
	std::vector<int> hist(256, 0);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			//hist.push_back(img.at<uchar>(i, j));
			hist[img.at<uchar>(i, j)]++;
		}
	}

	std::vector<float> pdf(256, 0);

	float sum = 0;
	for (int i = 0; i < 256; i++) {
			//hist.push_back(img.at<uchar>(i, j));
			pdf[i] = (float)hist[i] / (height * width);
			std::cout << pdf[i] << std::endl;
			sum += pdf[i];
	}

	std::cout << "Sum is: " << sum << std::endl;

	for (int i = 0; i < 1;) {
		if (waitKey() == 27)
			i++;
	}
}

void compute_for_bins(Mat& img, const int sizeOfBin) {
	int height = img.rows;
	int width = img.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	//histogram
	std::vector<int> hist(256, 0);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			hist[img.at<uchar>(i, j)]++;
		}
	}

	//compressed histogram
	std::vector<int> hist_bins((256/sizeOfBin), 0);
	float D = 256.0 / sizeOfBin;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			hist_bins[img.at<uchar>(i, j)/D]++;
		}
	}

	showHistogram("histogram image", &hist[0], 256, 256);
	showHistogram("histogram image with bins", &hist_bins[0], 256, 256);
	waitKey(0);
}

void multi_level_thresholding(Mat& img) {
	int height = img.rows;
	int width = img.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	//histogram
	std::vector<int> hist(256, 0);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			//hist.push_back(img.at<uchar>(i, j));
			hist[img.at<uchar>(i, j)]++;
		}
	}

	std::vector<float> pdf(256, 0);

	float sum = 0;
	for (int i = 0; i < 256; i++) {
		//hist.push_back(img.at<uchar>(i, j));
		pdf[i] = (float)hist[i] / (height * width);
		std::cout << pdf[i] << std::endl;
		sum += pdf[i];
	}

	int WH = 5;
	int windowWidth = (2 * WH) + 1;
	float threshold = 0.0003;


	std::vector<int> local_maximas(256, 0);
	local_maximas.push_back(0);

	for (int k = WH; k <= 255 - WH; k++) {
		float temp_sum = 0;
		float highest_in_window = pdf[k];

		for (int j = k-WH; j <= k+WH; j++) {
			temp_sum += pdf[j];

			if (pdf[j] > highest_in_window)
				highest_in_window = pdf[j];
		}

		float average_pdf = (float)temp_sum / windowWidth;

		std::cout << "Average pdf: " << average_pdf << " Current pdf: "<<pdf[k]<<" Highest in window: "<< highest_in_window<<std::endl;
		if ((pdf[k] > (average_pdf + threshold)) && pdf[k] == highest_in_window) {
			local_maximas.push_back(k);
			std::cout << "Pushed back "<< k << std::endl;
		}
	}
	local_maximas.push_back(255);

	for (int i = 0; i < local_maximas.size(); i++) {
		std::cout << local_maximas[i] << std::endl;
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			float difference = INT_MAX;
			float closest_maxima = 0;
			for (int k = 0; k < local_maximas.size(); k++) {
				if (abs(img.at<uchar>(i, j) - local_maximas[k]) < difference) {
					difference = img.at<uchar>(i, j) - local_maximas[k];
					closest_maxima = local_maximas[k];
				}
				//std::cout << "The smallest difference is: " << difference << " for value " << int(img.at<uchar>(i, j)) << " which is the local maxima " << closest_maxima << std::endl;
				dst.at<uchar>(i, j) = closest_maxima;
			}
		}
	}

	imshow("img", img);
	imshow("dst", dst);
	waitKey(0);

}



void floyd_steinberg_dithering(Mat& img) {
	int height = img.rows;
	int width = img.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	//histogram
	std::vector<int> hist(256, 0);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			//hist.push_back(img.at<uchar>(i, j));
			hist[img.at<uchar>(i, j)]++;
		}
	}

	std::vector<float> pdf(256, 0);

	float sum = 0;
	for (int i = 0; i < 256; i++) {
		//hist.push_back(img.at<uchar>(i, j));
		pdf[i] = (float)hist[i] / (height * width);
		std::cout << pdf[i] << std::endl;
		sum += pdf[i];
	}

	int WH = 5;
	int windowWidth = (2 * WH) + 1;
	float threshold = 0.0003;


	std::vector<int> local_maximas(256, 0);
	local_maximas.push_back(0);

	for (int k = WH; k <= 255 - WH; k++) {
		float temp_sum = 0;
		float highest_in_window = pdf[k];

		for (int j = k - WH; j <= k + WH; j++) {
			temp_sum += pdf[j];

			if (pdf[j] > highest_in_window)
				highest_in_window = pdf[j];
		}

		float average_pdf = (float)temp_sum / windowWidth;

		std::cout << "Average pdf: " << average_pdf << " Current pdf: " << pdf[k] << " Highest in window: " << highest_in_window << std::endl;
		if ((pdf[k] > (average_pdf + threshold)) && pdf[k] == highest_in_window) {
			local_maximas.push_back(k);
			std::cout << "Pushed back " << k << std::endl;
		}
	}
	local_maximas.push_back(255);

	for (int i = 0; i < local_maximas.size(); i++) {
		std::cout << local_maximas[i] << std::endl;
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			float oldpixel = img.at<uchar>(i, j);

			float difference = INT_MAX;
			float closest_maxima = 0;
			for (int k = 0; k < local_maximas.size(); k++) {
				if (abs(img.at<uchar>(i, j) - local_maximas[k]) < difference) {
					difference = img.at<uchar>(i, j) - local_maximas[k];
					closest_maxima = local_maximas[k];
				}
			}
			dst.at<uchar>(i, j) = closest_maxima;
		}
	}

	imshow("img", img);
	imshow("dst", dst);
	waitKey(0);

}

void driver_function_l3() {

	Mat img = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/Lab1/L1/PI-Images/Lena_24bits.bmp", IMREAD_COLOR);
	Mat img_gray = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/Lab1/L1/PI-Images/cameraman.bmp", IMREAD_GRAYSCALE);
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Exercise 1 compute histogram of grayscale image\n");
		printf(" 2 - Exercise 2 Compute probability density function from histogram\n");
		printf(" 3 - Exercise 3 Compute histogram for a number of bins\n");
		printf(" 4 - Exercise 4 Multi Level Thresholding\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			compute_histogram(img_gray);
			break;
		case 2:
			compute_pdf(img_gray);
			break;
		case 3:
			compute_for_bins(img_gray, 8);
			break;
		case 4:
			multi_level_thresholding(img_gray);
			break;
		}


	} while (op != 0);
}

void compute_object_properties_helper(int x, int y, Mat& img)
{
	Size img_size = Size(img.cols, img.rows);
	int height = img.rows;
	int width = img.cols;

	int color_R = (int)(img).at<Vec3b>(y, x)[2];
	int color_G = (int)(img).at<Vec3b>(y, x)[1];
	int color_B = (int)(img).at<Vec3b>(y, x)[0];

	int area = 0;
	int center_r = 0;
	int center_c = 0; 
	int num_of_pixels_on_perimeter = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			int color_R_temp = (int)(img).at<Vec3b>(i, j)[2];
			int color_G_temp = (int)(img).at<Vec3b>(i, j)[1];
			int color_B_temp = (int)(img).at<Vec3b>(i, j)[0];


			//if we are in the object
			if (color_R == color_R_temp && color_G == color_G_temp && color_B == color_B_temp)
			{
				area++; //caluclation of area
				center_c += j; //calculation for center of mass - column
				center_r += i; //calculation for center of mass - rows

				//compute perimeter
				//upper left
				int color_R_temp_1 = (int)(img).at<Vec3b>(i - 1, j - 1)[2];
				int color_G_temp_1 = (int)(img).at<Vec3b>(i - 1, j - 1)[1];
				int color_B_temp_1 = (int)(img).at<Vec3b>(i - 1, j - 1)[0];

				//upper middle
				int color_R_temp_2 = (int)(img).at<Vec3b>(i - 1, j)[2];
				int color_G_temp_2 = (int)(img).at<Vec3b>(i - 1, j)[1];
				int color_B_temp_2 = (int)(img).at<Vec3b>(i - 1, j)[0];

				//upper right
				int color_R_temp_3 = (int)(img).at<Vec3b>(i - 1, j + 1)[2];
				int color_G_temp_3 = (int)(img).at<Vec3b>(i - 1, j + 1)[1];
				int color_B_temp_3 = (int)(img).at<Vec3b>(i - 1, j + 1)[0];

				//middle left
				int color_R_temp_4 = (int)(img).at<Vec3b>(i, j - 1)[2];
				int color_G_temp_4 = (int)(img).at<Vec3b>(i, j - 1)[1];
				int color_B_temp_4 = (int)(img).at<Vec3b>(i, j - 1)[0];

				//middle right
				int color_R_temp_5 = (int)(img).at<Vec3b>(i, j + 1)[2];
				int color_G_temp_5 = (int)(img).at<Vec3b>(i, j + 1)[1];
				int color_B_temp_5 = (int)(img).at<Vec3b>(i, j + 1)[0];

				//bottom left
				int color_R_temp_6 = (int)(img).at<Vec3b>(i + 1, j - 1)[2];
				int color_G_temp_6 = (int)(img).at<Vec3b>(i + 1, j - 1)[1];
				int color_B_temp_6 = (int)(img).at<Vec3b>(i + 1, j - 1)[0];

				//bottom middle
				int color_R_temp_7 = (int)(img).at<Vec3b>(i + 1, j)[2];
				int color_G_temp_7 = (int)(img).at<Vec3b>(i + 1, j)[1];
				int color_B_temp_7 = (int)(img).at<Vec3b>(i + 1, j)[0];

				//bottom right
				int color_R_temp_8 = (int)(img).at<Vec3b>(i + 1, j + 1)[2];
				int color_G_temp_8 = (int)(img).at<Vec3b>(i + 1, j + 1)[1];
				int color_B_temp_8 = (int)(img).at<Vec3b>(i + 1, j + 1)[0];

				if ((color_R_temp_1 == 255 && color_G_temp_1 == 255 && color_B_temp_1 == 255) ||
					(color_R_temp_2 == 255 && color_G_temp_2 == 255 && color_B_temp_2 == 255) || 
					(color_R_temp_3 == 255 && color_G_temp_3 == 255 && color_B_temp_3 == 255) || 
					(color_R_temp_4 == 255 && color_G_temp_4 == 255 && color_B_temp_4 == 255) || 
					(color_R_temp_5 == 255 && color_G_temp_5 == 255 && color_B_temp_5 == 255) || 
					(color_R_temp_6 == 255 && color_G_temp_6 == 255 && color_B_temp_6 == 255) || 
					(color_R_temp_7 == 255 && color_G_temp_7 == 255 && color_B_temp_7 == 255) || 
					(color_R_temp_8 == 255 && color_G_temp_8 == 255 && color_B_temp_8 == 255))
				{
					num_of_pixels_on_perimeter++;
				}

			}


		}
	}

	center_r /= area;
	center_c /= area;

	int mrr = 0;
	int mcc = 0;
	int mrc = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			int color_R_temp = (int)(img).at<Vec3b>(i, j)[2];
			int color_G_temp = (int)(img).at<Vec3b>(i, j)[1];
			int color_B_temp = (int)(img).at<Vec3b>(i, j)[0];

			if (color_R == color_R_temp && color_G == color_G_temp && color_B == color_B_temp)
			{
				mrr += (i - center_r) * (i - center_r);
				mcc += (j - center_c) * (j - center_c);
				mrc += (i - center_r) * (j - center_c);
			}

		}
	}
	double phi = 0.5 * atan2((double)(2 * mrc), (double)(mcc - mrr));
	double deg = (phi + PI) * (180.0 / PI);

	num_of_pixels_on_perimeter = num_of_pixels_on_perimeter * PI / 4;

	float thickness_ratio = 4 * PI / (area / (num_of_pixels_on_perimeter * num_of_pixels_on_perimeter));

	printf("Area %d\nCenter of mass - row %d, column %d\nAngle of elongation - %f degrees\nPerimeter with 8 connectivity: %d\nThinness ratio: %f\n", area, center_r, center_c, deg, num_of_pixels_on_perimeter, thickness_ratio);
}

void compute_object_properties(Mat& img) {
	
	//Create a window
	namedWindow("My Window", 1);

	//set the callback function for any mouse event
	setMouseCallback("My Window", MyCallBackFunc, &img);

	//show the image
	imshow("My Window", img);

	// Wait until user press some key
	waitKey(0);

}

void driver_function_l4() {

	Mat img = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L4/PI-L4/trasaturi_geom.bmp", IMREAD_COLOR);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Exercise 1 Compute object characteristics and display\n");
		printf(" 2 - Exercise 2 Compute probability density function from histogram\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			compute_object_properties(img);
			break;
		case 2:
			break;
		}


	} while (op != 0);
}

Mat label_matrix_to_color_image(int** labels, int height, int width) {
	Mat dst = Mat(height, width, CV_8UC3, Scalar(0, 0, 0)); // Black background

	int maxLabel = 0;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (labels[i][j] > maxLabel)
				maxLabel = labels[i][j];

	// Generate random colors for each label
	std::vector<Vec3b> colors(maxLabel + 1, Vec3b(0, 0, 0)); // Background remains black
	for (int i = 1; i <= maxLabel; i++)
		colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256); // Random RGB

	// Assign colors to each pixel
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			dst.at<Vec3b>(i, j) = colors[labels[i][j]];

	return dst;
}

void bfs_labeling(Mat& img) {
	int height = img.rows;
	int width = img.cols;

	int label = 0;

	int dx[] = { -1, 1, 0, 0, -1, -1, 1, 1 };
	int dy[] = { 0, 0, -1, 1, -1, 1, -1, 1 };

	//allocate rows
	int** labels = (int**)malloc(height * sizeof(int*));
	if (!labels) {
		printf("Memory allocation failed\n");
		return;
	}


	//allocate columns
	for (int i = 0; i < height; i++) {
		labels[i] = (int*)malloc(width * sizeof(int));
		if (labels[i] == NULL) {
			printf("Memory allocation failed\n");
			return;
		}
	}

	//fill with 0 values (background)
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			labels[i][j] = label;
		}
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (img.at<uchar>(i, j) == 0 && labels[i][j] == 0) {
				label++;
				std::queue<std::pair<int, int>> Q;

				labels[i][j] = label;

				Q.push({i, j});


				while (!Q.empty()) {
					std::pair<int, int> p = Q.front();
					Q.pop();

					for (int d = 0; d < 8; d++) {  // 8 directions
						int ni = p.first + dx[d];
						int nj = p.second + dy[d];

						if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
							if (img.at<uchar>(ni, nj) == 0 && labels[ni][nj] == 0) {
								labels[ni][nj] = label;
								Q.push({ ni, nj });
							}
						}
					}

				}
			}
		}
	}

	Mat dst = label_matrix_to_color_image(labels, height, width);

	// Free allocated memory
	for (int i = 0; i < height; i++)
		free(labels[i]);
	free(labels);


	//show the image
	imshow("My Window", dst);

	// Wait until user press some key
	waitKey(0);


}

void two_pass_component_labeling(Mat& img) {
	int height = img.rows;
	int width = img.cols;

	int label = 0;

	//allocate rows
	int** labels = (int**)malloc(height * sizeof(int*));
	if (!labels) {
		printf("Memory allocation failed\n");
		return;
	}


	//allocate columns
	for (int i = 0; i < height; i++) {
		labels[i] = (int*)malloc(width * sizeof(int));
		if (labels[i] == NULL) {
			printf("Memory allocation failed\n");
			return;
		}
	}

	//fill with 0 values (background)
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			labels[i][j] = label;
		}
	}

	std::vector<std::vector<int>> edges(1000);

	int dx[] = { -1, 1, 0, 0, -1, -1, 1, 1 };
	int dy[] = { 0, 0, -1, 1, -1, 1, -1, 1 };

	// First Pass: Labeling & Equivalence Tracking
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (img.at<uchar>(i, j) == 0 && labels[i][j] == 0) { // Foreground pixel
				std::vector<int> L;

				// Check 4-connected neighbors
				for (int d = 0; d < 4; d++) {
					int ni = i + dx[d], nj = j + dy[d];
					if (ni >= 0 && ni < height && nj >= 0 && nj < width && labels[ni][nj] > 0) {
						L.push_back(labels[ni][nj]);
					}
				}

				if (L.empty()) {
					label++;
					labels[i][j] = label;
				}
				else {
					int min_label = *min_element(L.begin(), L.end());
					labels[i][j] = min_label;
					for (int y : L) {
						if (y != min_label) {
							edges[min_label].push_back(y);
							edges[y].push_back(min_label);
						}
					}
				}
			}
		}
	}

	// Display intermediate results (First Pass)
	Mat intermediate(height, width, CV_8UC3, Scalar(0, 0, 0));
	std::vector<Vec3b> colors(label + 1, Vec3b(0, 0, 0));

	for (int i = 1; i <= label; i++) {
		colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (labels[i][j] > 0) {
				intermediate.at<Vec3b>(i, j) = colors[labels[i][j]];
			}
		}
	}

	imshow("First Pass Labels", intermediate);
	waitKey(0);

	// Second Pass: Resolve Equivalences
	std::vector<int> newlabels(label + 1, 0);
	int newlabel = 0;

	for (int i = 1; i <= label; i++) {
		if (newlabels[i] == 0) {
			newlabel++;
			std::queue<int> Q;
			newlabels[i] = newlabel;
			Q.push(i);

			while (!Q.empty()) {
				int x = Q.front();
				Q.pop();
				for (int y : edges[x]) {
					if (newlabels[y] == 0) {
						newlabels[y] = newlabel;
						Q.push(y);
					}
				}
			}
		}
	}

	// Relabel image
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			labels[i][j] = newlabels[labels[i][j]];

	// Convert final label matrix to color image
	Mat final_img(height, width, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 1; i <= newlabel; i++) {
		colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (labels[i][j] > 0) {
				final_img.at<Vec3b>(i, j) = colors[labels[i][j]];
			}
		}
	}

	imshow("Final Labeled Components", final_img);
	waitKey(0);

	// Free memory
	for (int i = 0; i < height; i++)
		free(labels[i]);
	free(labels);

}
void driver_function_l5() {

	Mat img_gray = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L5/PI-L5/crosses.bmp", IMREAD_GRAYSCALE);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Exercise 1 Breadth first traversal component labeling algorithm\n");
		printf(" * - Exercise 2 Function which generates a color image from a label matrix (implemented but not useable alone)\n");
		printf(" 3 - Exercise 3 Two-pass component labeling algorithm\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			bfs_labeling(img_gray);
			break;
		case 2:
			break;
		case 3:
			two_pass_component_labeling(img_gray);
			break;
		}


	} while (op != 0);
}

void border_tracing(Mat& img) {
	int height = img.rows;
	int width = img.cols;

	int starting_i = -1;
	int starting_j = -1;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (img.at<uchar>(i, j) == 0) {
				starting_i = i;
				starting_j = j;
				goto found_start;
			}
		}
	}

	cout << "No foreground pixel found!" << endl;
	return;

found_start:
	int dir = 7;
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	Point p0(starting_j, starting_i);
	Point pk = p0;
	vector<Point> border;
	vector<int> chain;
	vector<int> derivatives;
	border.push_back(p0);
	Point p1;

	Mat dst(height, width, CV_8UC1, Scalar(255, 255, 255));

	while (border.size() <= 2 || p0 != border[border.size() - 2] || p1 != pk) {
		dir = (dir % 2 == 0) ? (dir + 7) % 8 : (dir + 6) % 8;
		Point next_pt = pk + Point(dj[dir], di[dir]);

		while (next_pt.y >= 0 && next_pt.y < height && next_pt.x >= 0 && next_pt.x < width &&
			img.at<uchar>(next_pt) == 255) {
			dir = (dir + 1) % 8;
			next_pt = pk + Point(dj[dir], di[dir]);
		}

		pk = next_pt;
		border.push_back(pk);
		chain.push_back(dir);
		dst.at<uchar>(pk) = 0;

		if (chain.size() > 1) {
			int cck = chain.back();
			int cck_1 = chain[chain.size() - 2];
			int derivative = (cck - cck_1 + 8) % 8;
			derivatives.push_back(derivative);
		}

		if (border.size() == 2) {
			p1 = pk;
		}
	}

	if (border.size() > 2) {
		border.erase(border.end() - 2, border.end());
	}
	if (chain.size() > 2) {
		chain.erase(chain.end() - 2, chain.end());
	}

	cout << "Border points:" << endl;
	for (const auto& point : border) {
		cout << "(" << point.x << ", " << point.y << ")\n";
	}

	cout << "Chain code:" << endl;
	for (int code : chain) {
		cout << code << " ";
	}
	cout << endl;

	cout << "Derivatives:" << endl;
	for (int derivative : derivatives) {
		cout << derivative << " ";
	}
	cout << endl;

	imshow("Final Border", dst);
	waitKey(0);
}

void draw_from_chain_code(Mat& img) {
	String text_path = "C:/Personal stuff/Year 3 Sem 2/IP/Labs/L6/PI-L6/reconstruct.txt";

	ifstream file(text_path);
	if (!file.is_open()) {
		cout << "Could not open the file!" << endl;
		return;
	}

	int start_x, start_y;
	file >> start_x >> start_y;

	int num_border_points;
	file >> num_border_points;

	vector<int> chain_code;
	int direction;
	while (file >> direction) {
		chain_code.push_back(direction);
	}

	file.close();

	if (chain_code.empty()) {
		cout << "Chain code is empty!" << endl;
		return;
	}

	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	Point current(start_x, start_y);

	for (size_t i = 0; i < chain_code.size(); i++) {
		int dir = chain_code[i];

		current.x += dj[dir];
		current.y += di[dir];

		if (current.x >= 0 && current.x < img.cols && current.y >= 0 && current.y < img.rows) {
			img.at<Vec3b>(current.y, current.x) = Vec3b(0, 0, 0);
		}
	}

	imshow("Reconstructed Border", img);
	waitKey(0);
}

void driver_function_l6() {

	Mat img_gray = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L6/PI-L6/triangle_up.bmp", IMREAD_GRAYSCALE);
	Mat img_gray_bg = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L6/PI-L6/gray_background.bmp", IMREAD_COLOR);
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Exercise 1 Border tracing and draw image\n");
		printf(" * - Exercise 2 Print chain code\n");
		printf(" 3 - Exercise 3 reconstruct the chain code\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			border_tracing(img_gray);
			break;
		case 2:
			break;
		case 3:
			draw_from_chain_code(img_gray_bg);
			break;
		}


	} while (op != 0);
}

Mat dilation(Mat& src) {
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1, Scalar(255));

	int dx[] = { -1, 0, 1, 0 };
	int dy[] = { 0, -1, 0, 1 };

	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			if (src.at<uchar>(i, j) == 0) {


				for (int d = 0; d < 4; d++) {

					dst.at<uchar>(i + dx[d], j + dy[d]) = 0;
				}
				dst.at<uchar>(i, j) = 0;
			}
		}
	}

	return dst;
}

void dilation_driver(int n_times) {
	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L7/PI-L7/Morphological_Op_Images/1_Dilate/mon1thr1_bw.bmp", IMREAD_GRAYSCALE);

	Mat dst = Mat(src);
	for (int i = 0; i < n_times; i++) {
		dst = dilation(dst);
	}

	imshow("dest", dst);
	imshow("source", src);
	waitKey(0);
}

Mat erosion(Mat& src) {
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1, Scalar(255));

	int dx[] = { -1, 0, 1, 0 };
	int dy[] = { 0, -1, 0, 1 };

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (src.at<uchar>(i, j) == 0) {

				bool flag = true;
				for (int d = 0; d < 4; d++) {
					if (src.at<uchar>(i + dx[d], j + dy[d]) != 0) {
						flag = false;
					}
				}
				if(flag)
					dst.at<uchar>(i, j) = 0;
			}
		}
	}

	return dst;
}

void erosion_driver(int n_times) {
	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L7/PI-L7/Morphological_Op_Images/2_Erode/mon1thr1_bw.bmp", IMREAD_GRAYSCALE);

	Mat dst = Mat(src);
	for (int i = 0; i < n_times; i++) {
		dst = erosion(dst);
	}

	imshow("dest", dst);
	imshow("source", src);
	waitKey(0);
}

void opening_driver(int n_times) {
	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L7/PI-L7/Morphological_Op_Images/3_Open/cel4thr3_bw.bmp", IMREAD_GRAYSCALE);

	Mat dst = Mat(src);
	for (int i = 0; i < n_times; i++) {
		dst = erosion(dst);
		dst = dilation(dst);
	}

	imshow("dest", dst);
	imshow("source", src);
	waitKey(0);
}
void closing_driver(int n_times) {
	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L7/PI-L7/Morphological_Op_Images/4_Close/art4_bw.bmp", IMREAD_GRAYSCALE);

	Mat dst = Mat(src);
	for (int i = 0; i < n_times; i++) {
		dst = dilation(dst);
		dst = erosion(dst);
	}

	imshow("dest", dst);
	imshow("source", src);
	waitKey(0);
}
void morphological_operations_driver(int n_times) {

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Exercise 1 Call Dilation\n");
		printf(" 2 - Exercise 1 Call Erosion\n");
		printf(" 3 - Exercise 3 Opening\n");
		printf(" 4 - Exercise 4 Closing\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			dilation_driver(n_times);
			break;
		case 2:
			erosion_driver(n_times);
			break;
		case 3:
			opening_driver(n_times);
			break;
		case 4:
			closing_driver(n_times);
			break;
		}


	} while (op != 0);

}
void morphological_operations_driver_n_time() {


	int op;
	int n_times;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Exercise 1 Call Dilation\n");
		printf(" 2 - Exercise 1 Call Erosion\n");
		printf(" 3 - Exercise 3 Opening\n");
		printf(" 4 - Exercise 4 Closing\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			printf(" Introduce how many times do you want to call this operation?\n");
			printf("Option: ");
			scanf("%d", &n_times);
			dilation_driver(n_times);
			break;
		case 2:
			printf(" Introduce how many times do you want to call this operation?\n");
			printf("Option: ");
			scanf("%d", &n_times);
			erosion_driver(n_times);
			break;
		case 3:
			printf(" Introduce how many times do you want to call this operation?\n");
			printf("Option: ");
			scanf("%d", &n_times);
			opening_driver(n_times);
			break;
		case 4:
			printf(" Introduce how many times do you want to call this operation?\n");
			printf("Option: ");
			scanf("%d", &n_times);
			closing_driver(n_times);
			break;
		}


	} while (op != 0);
}

void boundary_extraction() {
	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L7/PI-L7/Morphological_Op_Images/5_BoundaryExtraction/wdg2thr3_bw.bmp", IMREAD_GRAYSCALE);

	int height = src.rows;
	int width = src.cols;

	Mat temp = Mat(height, width, CV_8UC1, Scalar(255));
	Mat dst = Mat(height, width, CV_8UC1, Scalar(255));

	int dx[] = { -1, 1, 0, 0, -1, -1, 1, 1 };
	int dy[] = { 0, 0, -1, 1, -1, 1, -1, 1 };

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (src.at<uchar>(i, j) == 0) {

				bool flag = true;
				for (int d = 0; d < 8; d++) {
					if (src.at<uchar>(i + dx[d], j + dy[d]) != 0) {
						flag = false;
					}
				}
				if (flag)
					temp.at<uchar>(i, j) = 0;
			}
		}
	}

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if ((src.at<uchar>(i, j) == 0) && (temp.at<uchar>(i, j) != 0)) {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}

	imshow("dest", dst);
	imshow("temp", temp);
	imshow("source", src);
	waitKey(0);
}

void region_filling() {
	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L7/PI-L7/Morphological_Op_Images/6_RegionFilling/reg1neg1_bw.bmp", IMREAD_GRAYSCALE);

	int height = src.rows;
	int width = src.cols;

	Mat src_inv(height, width, CV_8UC1, Scalar(255));
	Mat temp(height, width, CV_8UC1, Scalar(255));
	Mat dst(height, width, CV_8UC1, Scalar(255));


	int dx[] = { -1, 1, 0, 0 };
	int dy[] = { 0, 0, -1, 1 };


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			src_inv.at<uchar>(i, j) = (src.at<uchar>(i, j) == 0) ? 255 : 0;
		}
	}


	temp.at<uchar>(height / 2, width / 2) = 0;


	bool outer_flag = true;
	while (outer_flag) {
		outer_flag = false;

		Mat next_temp = temp.clone();

		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				if (temp.at<uchar>(i, j) == 0) {
					for (int d = 0; d < 4; d++) {
						int ni = i + dx[d];
						int nj = j + dy[d];


						if (src_inv.at<uchar>(ni, nj) == 0 && src.at<uchar>(ni, nj) != 255) {
							next_temp.at<uchar>(ni, nj) = 0;
							outer_flag = true;
						}
					}
				}
			}
		}

		temp = next_temp;
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0 || temp.at<uchar>(i, j) == 0) {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}

	imshow("Source", src);
	imshow("Inverted", src_inv);
	imshow("Filled Region (temp)", temp);
	imshow("Final Output (dst)", dst);
	waitKey(0);
}

void driver_function_l7() {

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Exercise 1 Call Morphological Operations\n");
		printf(" 2 - Exercise 1 Call Morphological Operations n times\n");
		printf(" 3 - Exercise 3 Boundary Extraction\n");
		printf(" 4 - Exercise 4 Region Filling\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			morphological_operations_driver(1);
			break;
		case 2:
			morphological_operations_driver_n_time();
			break;
		case 3:
			boundary_extraction();
			break;
		case 4:
			region_filling();
			break;
		}


	} while (op != 0);
}

void l8_e1() {
	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L8/PI-L8/balloons.bmp", IMREAD_GRAYSCALE);

	int height = src.rows;
	int width = src.cols;
	int M = height * width;

	//histogram
	std::vector<int> hist(256, 0);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uchar pixel = src.at<uchar>(i, j);
			hist[pixel]++;
		}
	}

	//mean
	double mean = 0.0;
	for (int g = 0; g < 256; g++) {
		mean += g * hist[g];
	}
	mean /= M;

	//standard deviation
	double std_dev = 0.0;
	for (int g = 0; g < 256; g++) {
		std_dev += (g - mean) * (g - mean) * hist[g];
	}
	std_dev = std::sqrt(std_dev / M);

	//cumulative histogram
	std::vector<int> cum_hist(256, 0);
	cum_hist[0] = hist[0];
	for (int g = 1; g < 256; g++) {
		cum_hist[g] = cum_hist[g - 1] + hist[g];
	}

	
	cout << "Mean intensity value: " << mean << std::endl;
	cout << "Standard deviation: " << std_dev << std::endl;

	showHistogram("Histogram", &hist[0], 256, 256);
	showHistogram("Cumulative Histogram", &cum_hist[0], 256, 256);

	imshow("My Window", src);
	waitKey(0);
}

void automatic_threshold() {
	Mat img = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/Lab1/OpenCVApplication-VS2022_OCV490_basic/Images/eight.bmp", IMREAD_GRAYSCALE);

	int height = img.rows;
	int width = img.cols;
	int M = height * width;

	//histogram
	std::vector<int> hist(256, 0);
	int Imin = 255, Imax = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int g = img.at<uchar>(i, j);
			hist[g]++;
			if (g < Imin) Imin = g;
			if (g > Imax) Imax = g;
		}
	}

	//Initial threshold
	double T_prev = 0;
	double T = (Imin + Imax) / 2.0;

	//Iteratively compute new thresholds until convergence
	const double error = 1.0;
	while (std::abs(T - T_prev) >= error) {
		T_prev = T;

		// G1: [Imin, T], G2: (T, Imax]
		double sum1 = 0, count1 = 0, sum2 = 0, count2 = 0;
		for (int g = Imin; g <= (int)T; g++) {
			sum1 += g * hist[g];
			count1 += hist[g];
		}
		for (int g = (int)T + 1; g <= Imax; g++) {
			sum2 += g * hist[g];
			count2 += hist[g];
		}

		double mu1 = (count1 == 0) ? 0 : sum1 / count1;
		double mu2 = (count2 == 0) ? 0 : sum2 / count2;

		T = (mu1 + mu2) / 2.0;
	}

	std::cout << "Automatic threshold T: " << T << std::endl;

	//Threshold the image
	Mat thresholded = Mat::zeros(height, width, CV_8UC1);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			thresholded.at<uchar>(i, j) = (img.at<uchar>(i, j) > T) ? 255 : 0;
		}
	}
	imshow("Original Image", img);
	imshow("Thresholded Image", thresholded);
	waitKey(0);
}

Mat histogramStretchShrink(const Mat& img) {

	// Compute histogram
	vector<int> hist(256, 0);
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
			hist[img.at<uchar>(i, j)]++;

	// Auto-compute ginMin and ginMax
	int ginMin = 0, ginMax = 255;
	for (; ginMin < 256 && hist[ginMin] == 0; ++ginMin);
	for (; ginMax >= 0 && hist[ginMax] == 0; --ginMax);

	cout << "Detected ginMin: " << ginMin << ", ginMax: " << ginMax << endl;

	int goutMin, goutMax;
	cout << "Enter goutMin and goutMax for stretching/shrinking: ";
	cin >> goutMin >> goutMax;

	Mat result = img.clone();
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			int g = img.at<uchar>(i, j);
			g = max(min(g, ginMax), ginMin); // clip to [ginMin, ginMax]

			float gout = goutMin + (float)(g - ginMin) * (goutMax - goutMin) / (ginMax - ginMin);
			result.at<uchar>(i, j) = saturate_cast<uchar>(gout);
		}
	}

	// Compute histograms
	vector<int> hist_dst(256, 0);
	for (int i = 0; i < result.rows; ++i)
		for (int j = 0; j < result.cols; ++j)
			hist_dst[result.at<uchar>(i, j)]++;

	showHistogram("Histogram (Stretch/Shrink) - Source", &hist[0], 256, 256);
	showHistogram("Histogram (Stretch/Shrink) - Result", &hist_dst[0], 256, 256);
	imshow("Stretched/Shrunk Image", result);
	waitKey(0);
	return result;
}



Mat gammaCorrection(const Mat& img) {
	double gamma;
	cout << "Enter gamma coefficient: ";
	cin >> gamma;

	Mat result = img.clone();
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			float normalized = img.at<uchar>(i, j) / 255.0f;
			float corrected = pow(normalized, gamma);
			result.at<uchar>(i, j) = saturate_cast<uchar>(corrected * 255);
		}
	}

	vector<int> hist_src(256, 0), hist_dst(256, 0);
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j) {
			hist_src[img.at<uchar>(i, j)]++;
			hist_dst[result.at<uchar>(i, j)]++;
		}

	showHistogram("Histogram (Gamma) - Source", &hist_src[0], 256, 256);
	showHistogram("Histogram (Gamma) - Result", &hist_dst[0], 256, 256);
	imshow("Gamma Corrected Image", result);
	waitKey(0);
	return result;
}


Mat histogramSlide(const Mat& img) {
	int offset;
	cout << "Enter brightness offset: ";
	cin >> offset;

	Mat result = img.clone();
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			int g = img.at<uchar>(i, j) + offset;
			result.at<uchar>(i, j) = saturate_cast<uchar>(g);
		}
	}

	vector<int> hist_src(256, 0), hist_dst(256, 0);
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j) {
			hist_src[img.at<uchar>(i, j)]++;
			hist_dst[result.at<uchar>(i, j)]++;
		}

	showHistogram("Histogram (Slide) - Source", &hist_src[0], 256, 256);
	showHistogram("Histogram (Slide) - Result", &hist_dst[0], 256, 256);
	imshow("Brightness Adjusted Image", result);
	waitKey(0);
	return result;
}

void runHistogramTransforms() {
	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L8/PI-L8/balloons.bmp", IMREAD_GRAYSCALE);

	imshow("Original Image", src);

	histogramStretchShrink(src);
	gammaCorrection(src);
	histogramSlide(src);
}

Mat histogramEqualization() {
	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L8/PI-L8/balloons.bmp", IMREAD_GRAYSCALE);


	int totalPixels = src.rows * src.cols;
	std::vector<int> hist(256, 0);
	std::vector<float> pr(256, 0.0f);
	std::vector<float> pc(256, 0.0f);
	std::vector<uchar> tab(256, 0); 

	//histogram
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			uchar val = src.at<uchar>(i, j);
			hist[val]++;
		}
	}

	//PDF
	for (int i = 0; i < 256; ++i)
		pr[i] = static_cast<float>(hist[i]) / totalPixels;

	//CPDF
	pc[0] = pr[0];
	for (int i = 1; i < 256; ++i)
		pc[i] = pc[i - 1] + pr[i];

	//transformation table
	for (int i = 0; i < 256; ++i)
		tab[i] = saturate_cast<uchar>(255.0f * pc[i]);

	//Apply
	Mat dst = src.clone();
	for (int i = 0; i < src.rows; ++i)
		for (int j = 0; j < src.cols; ++j)
			dst.at<uchar>(i, j) = tab[src.at<uchar>(i, j)];

	//histograms for the output image
	std::vector<int> hist_dst(256, 0);
	for (int i = 0; i < dst.rows; ++i)
		for (int j = 0; j < dst.cols; ++j)
			hist_dst[dst.at<uchar>(i, j)]++;

	// Step 7: Show histograms and images
	showHistogram("Histogram - Source", &hist[0], 256, 256);
	showHistogram("Histogram - Equalized", &hist_dst[0], 256, 256);
	imshow("Original Image", src);
	imshow("Equalized Image", dst);
	waitKey(0);

	return dst;
}


void driver_function_l8() {

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Exercise 1 Deviation, histogram, and intensity levels\n");
		printf(" 2 - Exercise 1 Call Automatic Thresholding algorithm\n");
		printf(" 3 - Exercise 3 Histogram transformations\n");
		printf(" 4 - Exercise 4 Histogram Equalization\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			l8_e1();
			break;
		case 2:
			automatic_threshold();
			break;
		case 3:
			runHistogramTransforms();
			break;
		case 4:
			histogramEqualization();
			break;
		}


	} while (op != 0);
}

Mat applyCustomConvolutionFilter(const Mat& src, const Mat& kernel) {
	CV_Assert(src.type() == CV_8UC1); // Grayscale only
	CV_Assert(kernel.type() == CV_32FC1); // Kernel must be float

	int kRows = kernel.rows;
	int kCols = kernel.cols;
	int kCenterY = kRows / 2;
	int kCenterX = kCols / 2;

	Mat dst = Mat::zeros(src.size(), CV_32FC1); // float image for result

	// Compute scaling factor
	float sumPos = 0.0f, sumNeg = 0.0f, sumAll = 0.0f;
	for (int i = 0; i < kRows; ++i) {
		for (int j = 0; j < kCols; ++j) {
			float val = kernel.at<float>(i, j);
			sumAll += val;
			if (val > 0) sumPos += val;
			else sumNeg += -val;
		}
	}

	bool isLowPass = true;
	for (int i = 0; i < kRows; ++i) {
		for (int j = 0; j < kCols; ++j) {
			if (kernel.at<float>(i, j) < 0) {
				isLowPass = false;
				break;
			}
		}
	}

	float scale = 1.0f;
	if (isLowPass) {
		if (sumAll != 0)
			scale = 1.0f / sumAll;
	}
	else {
		float L = 255.0f;
		scale = L / (2.0f * max(sumPos, sumNeg)); // Option 3 from slide
	}

	// Apply convolution
	for (int y = kCenterY; y < src.rows - kCenterY; ++y) {
		for (int x = kCenterX; x < src.cols - kCenterX; ++x) {
			float acc = 0.0f;

			for (int i = 0; i < kRows; ++i) {
				for (int j = 0; j < kCols; ++j) {
					int yy = y + i - kCenterY;
					int xx = x + j - kCenterX;
					acc += src.at<uchar>(yy, xx) * kernel.at<float>(i, j);
				}
			}

			dst.at<float>(y, x) = acc * scale;
		}
	}

	// Normalize output for display
	Mat finalDst;
	dst.convertTo(finalDst, CV_8UC1, 1.0, 0); // Clip or normalize manually if needed

	return finalDst;
}

void driver_function_l9_e1() {

	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L9/PI-L9/cameraman.bmp", IMREAD_GRAYSCALE);

	Mat meanKernel = (Mat_<float>(3, 3) <<
		1, 1, 1,
		1, 1, 1,
		1, 1, 1);

	Mat gaussianKernel = (Mat_<float>(3, 3) <<
		1, 2, 1,
		2, 4, 2,
		1, 2, 1);

	Mat laplaceKernel = (Mat_<float>(3, 3) <<
		0, -1, 0,
		-1, 4, -1,
		0, -1, 0);

	Mat highPassKernel = (Mat_<float>(3, 3) <<
		-1, -1, -1,
		-1, 8, -1,
		-1, -1, -1);

	imshow("Original", src);
	imshow("Mean Filter", applyCustomConvolutionFilter(src, meanKernel));
	imshow("Gaussian Filter", applyCustomConvolutionFilter(src, gaussianKernel));
	imshow("Laplace Filter", applyCustomConvolutionFilter(src, laplaceKernel));
	imshow("High-Pass Filter", applyCustomConvolutionFilter(src, highPassKernel));

	waitKey(0);
}

void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

enum FilterType {
	IDEAL_LOW,
	IDEAL_HIGH,
	GAUSSIAN_LOW,
	GAUSSIAN_HIGH
};

Mat generic_frequency_domain_filter(Mat src, FilterType filterType)
{

	// Discrete Fourier Transform: https://docs.opencv.org/4.2.0/d8/d01/tutorial_discrete_fourier_transform.html
	int height = src.rows;
	int width = src.cols;

	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	// Centering transformation
	centering_transform(srcf);

	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	// the frequency is represented by its real and imaginary parts called frequency coefficients
	// split into real and imaginary channels fourier(i, j) = Re(i, j) + i * Im(i, j)
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);  // channels[0] = Re (real part), channels[1] = Im (imaginary part)

	//calculate magnitude and phase of the frequency by transforming it from cartesian to polar coordinates
	// the magnitude is useful for visualization

	Mat mag, phi;
	magnitude(channels[0], channels[1], mag); // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6d3b097586bca4409873d64a90fe64c3
	phase(channels[0], channels[1], phi); // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9db9ca9b4d81c3bde5677b8f64dc0137


	// TODO: Display here the log of magnitude (Add 1 to the magnitude to avoid log(0)) (see image 9.4e))
	// do not forget to normalize (you can use the normalize function from OpenCV)
	Mat logMag;
	log(mag + 1, logMag); // log(1 + magnitude)
	normalize(logMag, logMag, 0, 255, NORM_MINMAX);
	imshow("Log Magnitude Spectrum", logMag);
	waitKey(1);

	// TODO: Insert filtering operations here (channels[0] = Re(DFT(I), channels[1] = Im(DFT(I)); low pass or high pass filters
	// low pass filters equation 9.16 and equation 9.17
	// high pass filters equation 9.18 and 9.19

	int cx = width / 2;
	int cy = height / 2;

	float R = 20.0f; // Ideal cutoff radius
	float A = 10.0f; // Gaussian falloff

	// Create masks for filters
	Mat idealLow = Mat::zeros(src.size(), CV_32F);
	Mat idealHigh = Mat::zeros(src.size(), CV_32F);
	Mat gaussLow = Mat::zeros(src.size(), CV_32F);
	Mat gaussHigh = Mat::zeros(src.size(), CV_32F);

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			float D = sqrt(pow(i - cy, 2) + pow(j - cx, 2)); // Distance from center

			// Ideal
			idealLow.at<float>(i, j) = D <= R ? 1.0f : 0.0f;
			idealHigh.at<float>(i, j) = 1.0f - idealLow.at<float>(i, j);

			// Gaussian
			float gauss = exp(-(D * D) / (2 * A * A));
			gaussLow.at<float>(i, j) = gauss;
			gaussHigh.at<float>(i, j) = 1.0f - gauss;
		}
	}

	Mat* filterPtr;
	switch (filterType) {
	case IDEAL_LOW:    filterPtr = &idealLow; break;
	case IDEAL_HIGH:   filterPtr = &idealHigh; break;
	case GAUSSIAN_LOW: filterPtr = &gaussLow; break;
	case GAUSSIAN_HIGH:filterPtr = &gaussHigh; break;
	default:
		cout << "Invalid filter type provided!" << std::endl;
		return Mat(); // Return empty Mat to indicate failure
	}

	multiply(channels[0], *filterPtr, channels[0]);
	multiply(channels[1], *filterPtr, channels[1]);



	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT);

	// Inverse Centering transformation
	centering_transform(dstf);

	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

	return dst;
}

void driver_function_l9_e2()
{
	Mat img = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L9/PI-L9/cameraman.bmp", IMREAD_GRAYSCALE);


	vector<pair<FilterType, string>> filters = {
		{IDEAL_LOW,    "Ideal Low-Pass"},
		{IDEAL_HIGH,   "Ideal High-Pass"},
		{GAUSSIAN_LOW, "Gaussian Low-Pass"},
		{GAUSSIAN_HIGH,"Gaussian High-Pass"}
	};

	for (size_t i = 0; i < filters.size(); ++i) {
		FilterType type = filters[i].first;
		std::string name = filters[i].second;

		Mat result = generic_frequency_domain_filter(img, type);
		imshow(name, result);
	}

	waitKey(0);
	destroyAllWindows();
}


void driver_function_l9() {

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Exercise 1 Test mean filter, Gaussian filter, Laplace filters, High pass filters\n");
		printf(" 2 - Exercise 2  filtering in the frequency domain\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			driver_function_l9_e1();
			break;
		case 2:
			driver_function_l9_e2();
			break;
		}


	} while (op != 0);
}

Mat applyMedianFilter(const Mat src, int w) {

	Mat dst = src.clone();

	// Padding size
	int pad = w / 2;

	// Start timing
	double t = (double)getTickCount();

	for (int y = pad; y < src.rows - pad; ++y) {
		for (int x = pad; x < src.cols - pad; ++x) {
			vector<uchar> neighborhood;

			for (int i = -pad; i <= pad; ++i) {
				for (int j = -pad; j <= pad; ++j) {
					neighborhood.push_back(src.at<uchar>(y + i, x + j));
				}
			}

			// Sort and get median
			nth_element(neighborhood.begin(), neighborhood.begin() + neighborhood.size() / 2, neighborhood.end());
			dst.at<uchar>(y, x) = neighborhood[neighborhood.size() / 2];
		}
	}

	// End timing
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Median filter (" << w << "x" << w << ") processing time:  %.3f [ms]" << t * 1000 << endl;

	return dst;
}

void driver_function_l10_e1()
{
	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L9/PI-L9/cameraman.bmp", IMREAD_GRAYSCALE);

	int op;

	system("cls");
	destroyAllWindows();
	printf("Please enter the dimension of the filter (0 to exit):\n");
	scanf("%d", &op);
	Mat dst = applyMedianFilter(src, op);


	imshow("Initial image", src);
	imshow("Mean filtered image", dst);
	waitKey(0);
	destroyAllWindows();

}

Mat createGaussianKernel2D(int w, double sigma) {
	int k = w / 2;
	Mat kernel(w, w, CV_32FC1);

	for (int y = -k; y <= k; ++y) {
		for (int x = -k; x <= k; ++x) {
			float value = (1 / (2 * PI * sigma * sigma))*(exp(-(x * x + y * y) / (2 * sigma * sigma)));
			kernel.at<float>(y + k, x + k) = value;
		}
	}

	return kernel;
}

Mat applyGaussianFilter(const Mat src, int w) {


	int pad = w / 2;
	double sigma = max(w / 6.0, 1.0);

	Mat kernel = createGaussianKernel2D(w, sigma);
	Mat dst = Mat(src.size(), CV_8UC1);

	// Start timing
	double t = (double)getTickCount();

	for (int y = pad; y < src.rows - pad; ++y) {
		for (int x = pad; x < src.cols - pad; ++x) {
			float acc = 0.0f;

			for (int i = -pad; i <= pad; ++i) {
				for (int j = -pad; j <= pad; ++j) {
					uchar pixel = src.at<uchar>(y + i, x + j);
					float weight = kernel.at<float>(i + pad, j + pad);
					acc += pixel * weight;
				}
			}

			dst.at<uchar>(y, x) = acc;
		}
	}

	// End timing
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Gaussian filter (" << w << "x" << w << ") processing time: " << t * 1000 << " [ms]" << endl;

	return dst;
}

void driver_function_l10_e2()
{
	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L9/PI-L9/cameraman.bmp", IMREAD_GRAYSCALE);

	int op;

	system("cls");
	destroyAllWindows();
	printf("Please enter the dimension of the filter (0 to exit):\n");
	scanf("%d", &op);
	Mat dst = applyGaussianFilter(src, op);


	imshow("Initial image", src);
	imshow("Mean filtered image", dst);
	waitKey(0);
	destroyAllWindows();

}

Mat createGaussianKernel1D(int w, double sigma) {
	int k = w / 2;
	Mat kernel(1, w, CV_32FC1);

	for (int x = -k; x <= k; ++x) {
		float value = (1 / (sqrt(2 * PI) * sigma)) * exp(-(x * x) / (2 * sigma * sigma));
		kernel.at<float>(0, x + k) = value;
	}

	return kernel;
}


Mat applyGaussianFilter1D(const Mat src, int w) {
	int pad = w / 2;
	double sigma = w / 6.0;

	Mat gx = createGaussianKernel1D(w, sigma);
	Mat gy = gx.t();

	Mat temp = Mat(src.size(), CV_32FC1);
	Mat dst = Mat(src.size(), CV_8UC1);

	// Start timing
	double t = (double)getTickCount();

	//horizontal convolution
	for (int y = pad; y < src.rows - pad; ++y) {
		for (int x = pad; x < src.cols - pad; ++x) {
			float acc = 0.0f;
			for (int i = -pad; i <= pad; ++i) {
				uchar pixel = src.at<uchar>(y, x + i);
				float weight = gx.at<float>(0, i + pad);
				acc += pixel * weight;
			}
			temp.at<float>(y, x) = acc;
		}
	}

	//vertical convolution
	for (int y = pad; y < src.rows - pad; ++y) {
		for (int x = pad; x < src.cols - pad; ++x) {
			float acc = 0.0f;
			for (int i = -pad; i <= pad; ++i) {
				float pixel = temp.at<float>(y + i, x);
				float weight = gy.at<float>(i + pad, 0);
				acc += pixel * weight;
			}
			dst.at<uchar>(y, x) = saturate_cast<uchar>(acc);
		}
	}

	// End timing
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "1D Gaussian filter (" << w << "x" << w << ") processing time: " << t * 1000 << " [ms]" << endl;

	return dst;
}


void driver_function_l10_e3()
{
	Mat src = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L9/PI-L9/cameraman.bmp", IMREAD_GRAYSCALE);

	int op;

	system("cls");
	destroyAllWindows();
	printf("Please enter the dimension of the filter (0 to exit):\n");
	scanf("%d", &op);
	Mat dst = applyGaussianFilter1D(src, op);


	imshow("Initial image", src);
	imshow("Mean filtered image", dst);
	waitKey(0);
	destroyAllWindows();

}

void driver_function_l10() {

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Exercise 1  Median filter with a variable dimension \n");
		printf(" 2 - Exercise 2  Gaussian filter (matrix kernel) with a variable dimension \n");
		printf(" 3 - Exercise 3  Gaussian filter (1D vector kernel) with a variable dimension \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			driver_function_l10_e1();
			break;
		case 2:
			driver_function_l10_e2();
			break;
		case 3:
			driver_function_l10_e3();
			break;
		}


	} while (op != 0);
}

Mat create_fx() {
	Mat fx = Mat(3, 3, CV_32SC1);

	fx.at<int>(0, 0) = -1;
	fx.at<int>(0, 1) = 0;
	fx.at<int>(0, 2) = 1;

	fx.at<int>(1, 0) = -2;
	fx.at<int>(1, 1) = 0;
	fx.at<int>(1, 2) = 2;

	fx.at<int>(2, 0) = -1;
	fx.at<int>(2, 1) = 0;
	fx.at<int>(2, 2) = 1;

	return fx;
}

Mat create_fy() {
	Mat fy = Mat(3, 3, CV_32SC1);

	fy.at<int>(0, 0) = 1;
	fy.at<int>(0, 1) = 2;
	fy.at<int>(0, 2) = 1;

	fy.at<int>(1, 0) = 0;
	fy.at<int>(1, 1) = 0;
	fy.at<int>(1, 2) = 0;

	fy.at<int>(2, 0) = -1;
	fy.at<int>(2, 1) = -2;
	fy.at<int>(2, 2) = -1;

	return fy;
}

Mat applyConvolutionFilterWithoutScaling(const Mat& src, const Mat& kernel) {
	int kRows = kernel.rows;
	int kCols = kernel.cols;
	int kCenterY = kRows / 2;
	int kCenterX = kCols / 2;

	Mat dst = Mat::zeros(src.size(), CV_32SC1);

	// Apply convolution
	for (int y = kCenterY; y < src.rows - kCenterY; ++y) {
		for (int x = kCenterX; x < src.cols - kCenterX; ++x) {
			float acc = 0.0f;

			for (int i = 0; i < kRows; ++i) {
				for (int j = 0; j < kCols; ++j) {
					int yy = y + i - kCenterY;
					int xx = x + j - kCenterX;
					acc += src.at<uchar>(yy, xx) * kernel.at<int>(i, j);
				}
			}

			dst.at<float>(y, x) = acc;
		}
	}

	return dst;

}

Mat computeGradientMagnitude(const Mat& dx, const Mat& dy, Mat& direction) {
	Mat mag(dx.size(), CV_32FC1);
	direction = Mat(dx.size(), CV_32FC1);

	for (int y = 0; y < dx.rows; ++y) {
		for (int x = 0; x < dx.cols; ++x) {
			float gx = dx.at<float>(y, x);
			float gy = dy.at<float>(y, x);
			mag.at<float>(y, x) = sqrt(gx * gx + gy * gy);
			direction.at<float>(y, x) = atan2(gy, gx) + CV_PI; // shift from [-π, π] to [0, 2π]
		}
	}

	// Normalize to [0, 255]
	double maxVal;
	minMaxLoc(mag, nullptr, &maxVal);
	mag.convertTo(mag, CV_8UC1, 255.0 / maxVal);

	return mag;
}

Mat nonMaximaSuppression(const Mat& mag, const Mat& direction) {
	int rows = mag.rows;
	int cols = mag.cols;
	Mat suppressed = Mat::zeros(rows, cols, CV_8UC1);

	for (int y = 1; y < rows - 1; ++y) {
		for (int x = 1; x < cols - 1; ++x) {
			float angle = direction.at<float>(y, x) * 180.0 / CV_PI;  // Convert to degrees
			if (angle < 0) angle += 180;  // Normalize to [0,180)

			uchar currMag = mag.at<uchar>(y, x);
			uchar mag1 = 0, mag2 = 0;

			// Determine the direction sector and neighbors
			if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
				// 0° (horizontal)
				mag1 = mag.at<uchar>(y, x - 1);
				mag2 = mag.at<uchar>(y, x + 1);
			}
			else if (angle >= 22.5 && angle < 67.5) {
				// 45° (↙↗)
				mag1 = mag.at<uchar>(y - 1, x + 1);
				mag2 = mag.at<uchar>(y + 1, x - 1);
			}
			else if (angle >= 67.5 && angle < 112.5) {
				// 90° (vertical)
				mag1 = mag.at<uchar>(y - 1, x);
				mag2 = mag.at<uchar>(y + 1, x);
			}
			else if (angle >= 112.5 && angle < 157.5) {
				// 135° (↖↘)
				mag1 = mag.at<uchar>(y - 1, x - 1);
				mag2 = mag.at<uchar>(y + 1, x + 1);
			}

			if (currMag >= mag1 && currMag >= mag2) {
				suppressed.at<uchar>(y, x) = currMag;
			}
			else {
				suppressed.at<uchar>(y, x) = 0;
			}
		}
	}

	return suppressed;
}

Mat compute_magnitude(Mat& fx, Mat& fy) {
	Mat dst(fx.size(), CV_32FC1);

	for (int i = 0; i < fx.rows; ++i) {
		for (int j = 0; j < fx.cols; ++j) {
			float gx = fx.at<float>(i, j);
			float gy = fy.at<float>(i, j);
			dst.at<float>(i, j) = std::sqrt(gx * gx + gy * gy);
		}
	}

	// Normalize to 0–255 by dividing by 4√2
	dst /= (4.0f * std::sqrt(2.0f));

	// Convert to uchar for display
	Mat finalDst;
	dst.convertTo(finalDst, CV_8UC1);
	return finalDst;
}

Mat non_max_suppression(Mat& magnitude, Mat& fx, Mat& fy) {
	Mat suppressed = Mat::zeros(magnitude.size(), CV_8UC1);

	for (int y = 1; y < magnitude.rows - 1; ++y) {
		for (int x = 1; x < magnitude.cols - 1; ++x) {
			float gx = fx.at<float>(y, x);
			float gy = fy.at<float>(y, x);
			float theta = atan2(gy, gx) + CV_PI; // [0, 2π]
			int dir = static_cast<int>(round(theta / (CV_PI / 4.0))) % 8;
			int quantized = dir % 4;

			uchar current = magnitude.at<uchar>(y, x);
			uchar neighbor1 = 0, neighbor2 = 0;

			switch (quantized) {
			case 0: // horizontal 
				neighbor1 = magnitude.at<uchar>(y, x - 1);
				neighbor2 = magnitude.at<uchar>(y, x + 1);
				break;
			case 1: // diagonal 
				neighbor1 = magnitude.at<uchar>(y - 1, x + 1);
				neighbor2 = magnitude.at<uchar>(y + 1, x - 1);
				break;
			case 2: // vertical 
				neighbor1 = magnitude.at<uchar>(y - 1, x);
				neighbor2 = magnitude.at<uchar>(y + 1, x);
				break;
			case 3: // diagonal inverse
				neighbor1 = magnitude.at<uchar>(y - 1, x - 1);
				neighbor2 = magnitude.at<uchar>(y + 1, x + 1);
				break;
			}

			if (current >= neighbor1 && current >= neighbor2) {
				suppressed.at<uchar>(y, x) = current;
			}
			else {
				suppressed.at<uchar>(y, x) = 0;
			}
		}
	}

	return suppressed;
}

Mat apply_sobel_filter(Mat& img, Mat sobel1) {



	int kRows = sobel1.rows;
	int kCols = sobel1.cols;
	int kCenterY = kRows / 2;
	int kCenterX = kCols / 2;

	Mat dst = Mat::zeros(img.size(), CV_32FC1); // float image for result


	float sumPos = 0.0f, sumNeg = 0.0f, sumAll = 0.0f;
	for (int i = 0; i < kRows; ++i) {
		for (int j = 0; j < kCols; ++j) {
			float val = sobel1.at<float>(i, j);
			sumAll += val;
			if (val > 0) sumPos += val;
			else sumNeg += -val;
		}
	}
	bool isLowPass = true;
	for (int i = 0; i < kRows; ++i) {
		for (int j = 0; j < kCols; ++j) {
			if (sobel1.at<float>(i, j) < 0) {
				isLowPass = false;
				break;
			}
		}
	}




	// Apply convolution
	for (int y = kCenterY; y < img.rows - kCenterY; ++y) {
		for (int x = kCenterX; x < img.cols - kCenterX; ++x) {
			float acc = 0.0f;

			for (int i = 0; i < kRows; ++i) {
				for (int j = 0; j < kCols; ++j) {
					int yy = y + i - kCenterY;
					int xx = x + j - kCenterX;
					acc += img.at<uchar>(yy, xx) * sobel1.at<float>(i, j);
				}
				dst.at<float>(y, x) = acc;
			}

		}
	}



	return dst;
}

Mat adaptiveThresholding(const Mat& suppressed) {

	Mat thresholded = Mat::zeros(suppressed.size(), CV_8UC1);

	int nonZeroPixels = 0;
	for (int y = 0; y < suppressed.rows; y++) {
		for (int x = 0; x < suppressed.cols; x++) {
			if (suppressed.at<uchar>(y, x) > 0) {
				nonZeroPixels++;
			}
		}
	}

	if (nonZeroPixels == 0) {
		return thresholded;
	}

	float p = 0.1f;
	int numEdgePixels = static_cast<int>(p * nonZeroPixels);

	std::vector<uchar> nonZeroGradients;
	for (int y = 0; y < suppressed.rows; y++) {
		for (int x = 0; x < suppressed.cols; x++) {
			uchar val = suppressed.at<uchar>(y, x);
			if (val > 0) {
				nonZeroGradients.push_back(val);
			}
		}
	}

	std::sort(nonZeroGradients.begin(), nonZeroGradients.end());

	int thresholdHigh = nonZeroGradients[nonZeroGradients.size() - numEdgePixels];

	int thresholdLow = static_cast<int>(0.4f * thresholdHigh);

	thresholdLow = max(thresholdLow, 20);
	thresholdHigh = max(thresholdHigh, 50);

	std::cout << "ThresholdLow: " << thresholdLow << ", ThresholdHigh: " << thresholdHigh << std::endl;

	for (int y = 0; y < suppressed.rows; y++) {
		for (int x = 0; x < suppressed.cols; x++) {
			uchar val = suppressed.at<uchar>(y, x);

			if (val < thresholdLow) {
				thresholded.at<uchar>(y, x) = 0;
			}
			else if (val >= thresholdLow && val <= thresholdHigh) {
				thresholded.at<uchar>(y, x) = 128;
			}
			else {
				thresholded.at<uchar>(y, x) = 255;
			}
		}
	}

	return thresholded;
}

Mat edgeLinkingHysteresis(const Mat& thresholded) {
	Mat linked = thresholded.clone();
	int rows = linked.rows;
	int cols = linked.cols;

	int dx[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dy[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

	std::queue<cv::Point> q;

	for (int y = 1; y < rows - 1; y++) {
		for (int x = 1; x < cols - 1; x++) {
			if (linked.at<uchar>(y, x) == 255) {
				q.push(cv::Point(x, y));

				while (!q.empty()) {
					cv::Point p = q.front();
					q.pop();

					for (int i = 0; i < 8; i++) {
						int nx = p.x + dx[i];
						int ny = p.y + dy[i];


						if (nx >= 1 && nx < cols - 1 && ny >= 1 && ny < rows - 1) {
							if (linked.at<uchar>(ny, nx) == 128) {
								linked.at<uchar>(ny, nx) = 255;
								q.push(cv::Point(nx, ny));
							}
						}
					}
				}
			}
		}
	}

	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			if (linked.at<uchar>(y, x) == 128) {
				linked.at<uchar>(y, x) = 0;
			}
		}
	}

	return linked;
}

Mat improved_non_max_suppression(const Mat& magnitude, const Mat& fx, const Mat& fy) {
	Mat suppressed = Mat::zeros(magnitude.size(), CV_8UC1);
	int rows = magnitude.rows;
	int cols = magnitude.cols;

	for (int y = 1; y < rows - 1; y++) {
		for (int x = 1; x < cols - 1; x++) {
			float gx = fx.at<float>(y, x);
			float gy = fy.at<float>(y, x);

			float angle = atan2(gy, gx) * 180.0f / CV_PI;
			if (angle < 0) angle += 180.0f;

			uchar current = magnitude.at<uchar>(y, x);
			uchar neighbor1 = 0, neighbor2 = 0;

			if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
				neighbor1 = magnitude.at<uchar>(y, x - 1);
				neighbor2 = magnitude.at<uchar>(y, x + 1);
			}
			else if (angle >= 22.5 && angle < 67.5) {
				neighbor1 = magnitude.at<uchar>(y - 1, x + 1);
				neighbor2 = magnitude.at<uchar>(y + 1, x - 1);
			}
			else if (angle >= 67.5 && angle < 112.5) {
				neighbor1 = magnitude.at<uchar>(y - 1, x);
				neighbor2 = magnitude.at<uchar>(y + 1, x);
			}
			else if (angle >= 112.5 && angle < 157.5) {
				neighbor1 = magnitude.at<uchar>(y - 1, x - 1);
				neighbor2 = magnitude.at<uchar>(y + 1, x + 1);
			}

			if (current >= neighbor1 && current >= neighbor2) {
				suppressed.at<uchar>(y, x) = current;
			}
		}
	}

	return suppressed;
}

Mat revised_compute_magnitude(const Mat& fx, const Mat& fy) {
	Mat magnitude(fx.size(), CV_32FC1);

	for (int i = 0; i < fx.rows; ++i) {
		for (int j = 0; j < fx.cols; ++j) {
			float gx = fx.at<float>(i, j);
			float gy = fy.at<float>(i, j);
			magnitude.at<float>(i, j) = std::sqrt(gx * gx + gy * gy);
		}
	}

	double minVal, maxVal;
	minMaxLoc(magnitude, &minVal, &maxVal);

	Mat normalizedMagnitude;
	magnitude.convertTo(normalizedMagnitude, CV_8UC1, 255.0 / maxVal);

	return normalizedMagnitude;
}

Mat cannyEdgeDetection(const Mat& image) {
	Mat smoothed;
	GaussianBlur(image, smoothed, Size(5, 5), 1.4);

	Mat sobelX = (Mat_<float>(3, 3) <<
		- 1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);

	Mat sobelY = (Mat_<float>(3, 3) <<
		1, 2, 1,
		0, 0, 0,
		-1, -2, -1);

	Mat fx = apply_sobel_filter(smoothed, sobelX);
	Mat fy = apply_sobel_filter(smoothed, sobelY);

	Mat magnitude = revised_compute_magnitude(fx, fy);

	Mat suppressed = improved_non_max_suppression(magnitude, fx, fy);

	Mat thresholded = adaptiveThresholding(suppressed);

	Mat edges = edgeLinkingHysteresis(thresholded);

	return edges;
}

void driver_function_l11() {
	Mat img = imread("C:/Personal stuff/Year 3 Sem 2/IP/Labs/L1/PI-Images/saturn.bmp", IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cout << "Error loading image!" << std::endl;
		return;
	}

	Mat edges = cannyEdgeDetection(img);

	imshow("Original", img);

	Mat smoothed = applyGaussianFilter(img, 3.5);
	imshow("Gaussian Filtered", smoothed);

	Mat sobelX = (Mat_<float>(3, 3) <<
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);

	Mat sobelY = (Mat_<float>(3, 3) <<
		1, 2, 1,
		0, 0, 0,
		-1, -2, -1);

	Mat fx = apply_sobel_filter(smoothed, sobelX);
	Mat fy = apply_sobel_filter(smoothed, sobelY);
	Mat magnitude = compute_magnitude(fx, fy);
	imshow("Gradient Magnitude", magnitude);

	Mat suppressed = non_max_suppression(magnitude, fx, fy);
	imshow("Non-Maximum Suppression", suppressed);

	Mat thresholded = adaptiveThresholding(suppressed);
	imshow("Thresholded", thresholded);

	imshow("Canny Edges", edges);

	waitKey(0);
}

int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - Lab 1 exercises\n");
		printf(" 14 - Lab 2 exercises\n");
		printf(" 15 - Lab 3 exercises\n");
		printf(" 16 - Lab 4 exercises\n");
		printf(" 17 - Lab 5 exercises\n");
		printf(" 18 - Lab 6 exercises\n");
		printf(" 19 - Lab 7 exercises\n");
		printf(" 20 - Lab 8 exercises\n");
		printf(" 21 - Lab 9 exercises\n");
		printf(" 22 - Lab 10 exercises\n");
		printf(" 23 - Lab 11 exercises\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				testBGR2HSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;
			case 13:
				driver_function_l1();
				break;
			case 14:
				driver_function_l2();
				break;
			case 15:
				driver_function_l3();
				break;
			case 16:
				driver_function_l4();
				break;
			case 17:
				driver_function_l5();
				break;
			case 18:
				driver_function_l6();
				break;
			case 19:
				driver_function_l7();
				break;
			case 20:
				driver_function_l8();
				break;
			case 21:
				driver_function_l9();
				break;
			case 22:
				driver_function_l10();
				break;
			case 23:
				driver_function_l11();
				break;
		}
	}
	while (op!=0);
	return 0;
}