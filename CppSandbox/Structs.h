#pragma once

#include <vector>
#include "CvInclude.h"

enum Gender
{
	Unknown = 0,
	Male = 1,
	Female = 2
};

typedef std::vector<std::vector<float>> Anchor;
typedef std::vector<float> FaceIndex;
typedef std::pair<Gender, int> GenderAgeAttributes;

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
	cv::Mat normImage;
	FaceIndex index;
	std::string label;
	float similarity;
	Gender gender;
	int age;
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