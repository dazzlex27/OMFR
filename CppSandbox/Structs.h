#pragma once

#include <vector>
#include "CvInclude.h"

typedef std::vector<std::vector<float>> Anchor;

struct FaceDetectionResult
{
	std::vector<float> scores;
	std::vector<cv::Rect2f> boxes;
	std::vector<Landmarks> landmarks;
};

struct Face
{
	cv::Rect2f box;
	float score;
	Landmarks landmarks;
};

struct AnchorKey
{
	int width;
	int height;
	int stride;
};

inline bool operator<(const AnchorKey& l, const AnchorKey& r)
{
	return (l.width < r.width || (l.height == r.height && l.stride < r.stride));
}