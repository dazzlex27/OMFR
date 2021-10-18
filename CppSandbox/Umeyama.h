#pragma once

#include "CvInclude.h"

class Umeyama
{
public:
	cv::Mat GetSimilarTransform(const cv::Mat& src, const cv::Mat& dst) const;

private:
	const cv::Mat MeanAxis0(const cv::Mat& src) const;
	const cv::Mat ElementwiseMinus(const cv::Mat& m1, const cv::Mat& m2) const;
	const cv::Mat VarAxis0(const cv::Mat& src) const;
	const int MatrixRank(const cv::Mat& m) const;
};