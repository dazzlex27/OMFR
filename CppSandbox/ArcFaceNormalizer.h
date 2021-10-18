#pragma once

#include "Structs.h"
#include "CvInclude.h"
#include "Umeyama.h"

class ArcFaceNormalizer
{
private:
	const cv::Size _lmArraySize = cv::Size(2, 5);
	const std::vector<float> _dstMap = { 0.34191, 0.46157, 0.65653, 0.45983, 0.50022, 0.64050, 0.37097, 0.82469, 0.63151, 0.82325 };
	//float dstMap[lmCount * lmPoints] = { 38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041 };
	Umeyama _transformer;

public:
	std::vector<cv::Mat> GetNormalizedFaces(const cv::Mat& image, const std::vector<Face>& faces) const;
};