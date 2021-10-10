#include <opencv2/opencv.hpp>

int main()
{
	cv::Mat img = cv::imread("sh.jpg");
	cv::imshow("output", img);
	cv::waitKey(0);
}