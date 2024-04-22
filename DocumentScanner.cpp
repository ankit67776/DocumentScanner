#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat imgOriginal, imgGray, imgCrop, imgCanny, imgThre, imgDil, imgErode, imgBlur, imgWarp;
vector<Point> initialPoints, docPoints;

float w = 420, h = 596;

Mat preProcessing(Mat img) {
	
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgBlur, imgCanny, 25, 75);

	Mat kernal = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernal);
	//erode(imgDil, imgErode, kernal);

	return imgDil;
}



vector<Point> getContours(Mat image) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	vector<Point> biggest;
	int maxArea = 0;

	
	for (int j = 0; j < contours.size(); j++) {
		int area = contourArea(contours[j]);
		//cout << area << endl;

		string objectType;
		if (area > 1000) {

			float peri = arcLength(contours[j], true);
			approxPolyDP(contours[j], conPoly[j], 0.02 * peri, true);

			if (area > maxArea && conPoly[j].size() == 4) {
				//drawContours(imgOriginal, conPoly, j, Scalar(255, 0, 255), 5);
				biggest = { conPoly[j][0], conPoly[j][1], conPoly[j][2], conPoly[j][3] }; 
				maxArea = area;

			}

			//drawContours(imgOriginal, conPoly, j, Scalar(255, 0, 255), 5);
			//rectangle(imgOriginal, boundRect[j].tl(), boundRect[j].br(), Scalar(0, 255, 0), 5);

		}
	}
	return biggest;
}


void drawPoints(vector<Point> points, Scalar color) {

	for (int j = 0; j < points.size(); j++) {
		circle(imgOriginal, points[j], 10, color, FILLED);
		putText(imgOriginal, to_string(j), points[j], FONT_HERSHEY_PLAIN, 2, color, 2);

	}

}


vector<Point> reorder(vector<Point> points) {
	vector<Point> newPoints;
	vector<int> sumPoints, subPoints;

	for (int j = 0; j < points.size(); j++) {
		sumPoints.push_back(points[j].x + points[j].y);
		subPoints.push_back(points[j].x - points[j].y);
	}

	newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // 0
	newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); // 1
	newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); // 2
	newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // 3
	
	return newPoints;
}

Mat getWarp(Mat img, vector<Point> points, float w, float h) {

	Point2f src[4] = { points[0], points[1], points[2], points[3] };
	Point2f dst[4] = { {0.0f, 0.0f}, {w, 0.0f}, {0.0f, h}, {w, h} };

	Mat matrix = getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, matrix, Point(w, h));

	return imgWarp;
}


void main() {
	string path = "Resources/paper.jpg";
	imgOriginal = imread(path);

	resize(imgOriginal, imgOriginal, Size(), 0.5, 0.5);

	// preprocessing
	imgThre = preProcessing(imgOriginal);	

	// Get Contours - Biggest
	initialPoints = getContours(imgThre);
	//drawPoints(initialPoints, Scalar(0, 0, 255));
	docPoints = reorder(initialPoints);
	//drawPoints(docPoints, Scalar(0, 255, 0));

	// Warp
	imgWarp = getWarp(imgOriginal, docPoints, w, h);

	// Crop
	Rect roi(5, 5, w - (2 * 5), h - (2 * 5));
	imgCrop = imgWarp(roi);

	imshow("Image", imgOriginal);
	imshow("Image Dilation", imgThre);
	imshow("Image Warp", imgWarp);
	waitKey(0);
}