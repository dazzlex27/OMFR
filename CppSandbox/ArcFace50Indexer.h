#pragma once

#include "Structs.h"
#include <onnxruntime_cxx_api.h>

class ArcFace50Indexer
{
private:
	Ort::Session _session;
	const cv::Size _inputSize = cv::Size(112, 112);
	const int _inputDepth = 3;

public:
	ArcFace50Indexer(Ort::Env& env, const std::string& modelFilepath);
	FaceIndex GetIndex(const cv::Mat& faceImage);

private:
	cv::Mat PrepareImage(const cv::Mat& image, float* scaleFactor) const;
	FaceIndex RunNet(const cv::Mat& floatImage);
};