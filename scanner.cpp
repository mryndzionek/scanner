#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
	double i = fabs( contourArea(cv::Mat(contour1)) );
	double j = fabs( contourArea(cv::Mat(contour2)) );
	return ( i > j );
}

bool compareXCords(Point p1, Point p2)
{
	return (p1.x < p2.x);
}

bool compareYCords(Point p1, Point p2)
{
	return (p1.y < p2.y);
}

bool compareDistance(pair<Point, Point> p1, pair<Point, Point> p2)
{
	return (norm(p1.first - p1.second) < norm(p2.first - p2.second));
}

double _distance(Point p1, Point p2)
{
	return sqrt(((p1.x - p2.x) * (p1.x - p2.x)) +
			((p1.y  - p2.y) * (p1.y - p2.y)));
}

void resizeToHeight(Mat src, Mat &dst, int height)
{
	Size s = Size(src.cols * (height / double(src.rows)), height);
	resize(src, dst, s, CV_INTER_AREA);
}

void orderPoints(vector<Point> inpts, vector<Point> &ordered)
{
	sort(inpts.begin(), inpts.end(), compareXCords);
	vector<Point> lm(inpts.begin(), inpts.begin()+2);
	vector<Point> rm(inpts.end()-2, inpts.end());

	sort(lm.begin(), lm.end(), compareYCords);
	Point tl(lm[0]);
	Point bl(lm[1]);
	vector<pair<Point, Point> > tmp;
	for(size_t i = 0; i< rm.size(); i++)
	{
		tmp.push_back(make_pair(tl, rm[i]));
	}

	sort(tmp.begin(), tmp.end(), compareDistance);
	Point tr(tmp[0].second);
	Point br(tmp[1].second);

	ordered.push_back(tl);
	ordered.push_back(tr);
	ordered.push_back(br);
	ordered.push_back(bl);
}

void fourPointTransform(Mat src, Mat &dst, vector<Point> pts)
{
	vector<Point> ordered_pts;
	orderPoints(pts, ordered_pts);

	double wa = _distance(ordered_pts[2], ordered_pts[3]);
	double wb = _distance(ordered_pts[1], ordered_pts[0]);
	double mw = max(wa, wb);

	double ha = _distance(ordered_pts[1], ordered_pts[2]);
	double hb = _distance(ordered_pts[0], ordered_pts[3]);
	double mh = max(ha, hb);

	Point2f src_[] =
	{
			Point2f(ordered_pts[0].x, ordered_pts[0].y),
			Point2f(ordered_pts[1].x, ordered_pts[1].y),
			Point2f(ordered_pts[2].x, ordered_pts[2].y),
			Point2f(ordered_pts[3].x, ordered_pts[3].y),
	};
	Point2f dst_[] =
	{
			Point2f(0,0),
			Point2f(mw - 1, 0),
			Point2f(mw - 1, mh - 1),
			Point2f(0, mh - 1)
	};
	Mat m = getPerspectiveTransform(src_, dst_);
	warpPerspective(src, dst, m, Size(mw, mh));
}

void preProcess(Mat src, Mat &dst)
{
	cv::Mat imageGrayed;
	cv::Mat imageOpen, imageClosed, imageBlurred;

	cvtColor(src, imageGrayed, CV_BGR2GRAY);

	cv::Mat structuringElmt = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4,4));
	morphologyEx(imageGrayed, imageOpen, cv::MORPH_OPEN, structuringElmt);
	morphologyEx(imageOpen, imageClosed, cv::MORPH_CLOSE, structuringElmt);

	GaussianBlur(imageClosed, imageBlurred, Size(7, 7), 0);
	Canny(imageBlurred, dst, 75, 100);
}

string getOutputFileName(string path, string name)
{
	std::string fname, ext;

	size_t sep = path.find_last_of("\\/");
	if (sep != std::string::npos)
	{
		path = path.substr(sep + 1, path.size() - sep - 1);

		size_t dot = path.find_last_of(".");
		if (dot != std::string::npos)
		{
			fname = path.substr(0, dot);
			ext  = path.substr(dot, path.size() - dot);
		}
		else
		{
			fname = path;
			ext  = "";
		}
	}

	return fname + "_" + name + ext;
}

int main( int argc, char** argv)
{

	static const char * const keys = "{ i |image| }";
	CommandLineParser parser(argc, argv, keys);

	string image_name(parser.get<String>("image"));

	if (image_name.empty())
	{
	    parser.printParams();
	    return -1;
	}

	Mat image = imread(image_name);
	if (image.empty())
	{
		printf("Cannot read image file: %s\n", image_name.c_str());
		return -1;
	}

	double ratio = image.rows / 500.0;
	Mat orig = image.clone();
	resizeToHeight(image, image, 500);

	Mat gray, edged, warped;
	preProcess(image, edged);
#ifndef NDEBUG
	imwrite(getOutputFileName(image_name, "edged"), edged);
#endif

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<vector<Point> > approx;
	findContours(edged, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	approx.resize(contours.size());
	size_t i,j;
	for(i = 0; i< contours.size(); i++)
	{
		double peri = arcLength(contours[i], true);
		approxPolyDP(contours[i], approx[i], 0.02 * peri, true);
	}
	sort(approx.begin(), approx.end(), compareContourAreas);

	for(i = 0; i< approx.size(); i++)
	{
		drawContours(image, approx, i, Scalar(255, 255, 0), 2);
		if(approx[i].size() == 4)
		{
			break;
		}
	}

	if(i < approx.size())
	{
		drawContours(image, approx, i, Scalar(0, 255, 0), 2);
#ifndef NDEBUG
		imwrite(getOutputFileName(image_name, "outline"), image);
#endif
		for(j = 0; j< approx[i].size(); j++)
		{
			approx[i][j] *= ratio;
		}

		fourPointTransform(orig, warped, approx[i]);
#ifndef NDEBUG
		imwrite(getOutputFileName(image_name, "flat"), warped);
#endif
		cvtColor(warped, warped, CV_BGR2GRAY, 1);
		adaptiveThreshold(warped, warped, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 15);
		GaussianBlur(warped, warped, Size(3, 3), 0);
		imwrite(getOutputFileName(image_name, "scanned"), warped);
	}
}
