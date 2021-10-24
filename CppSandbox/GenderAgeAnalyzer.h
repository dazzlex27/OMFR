#pragma once

#include "Structs.h"
#include <onnxruntime_cxx_api.h>

class GenderAgeAnalyzer
{
private:
	Ort::Session _session;
	const cv::Size _inputSize = cv::Size(96,96);
	const int _inputDepth = 3;

public:
	GenderAgeAnalyzer(Ort::Env& env, const std::string& modelFilepath);
	GenderAgeAttributes GetAttributes(const cv::Mat& faceImage);

private:
	cv::Mat PrepareImage(const cv::Mat& image, float* scaleFactor) const;
	std::vector<float> RunNet(const cv::Mat& floatImage);
	GenderAgeAttributes GetResultFromTensorOutput(const std::vector<float>& tensorOutput) const;
};