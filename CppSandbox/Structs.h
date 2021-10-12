#pragma once

#include <vector>
#include "CvInclude.h"

typedef std::vector<std::vector<float>> Anchor;

struct FaceDetectionResult
{
	std::vector<float> scores;
	std::vector<cv::Rect2f> boxes;
	std::vector<Keypoints> keypoints;
};

struct Face
{
	cv::Rect2f box;
	float score;
	Keypoints keypoints;
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