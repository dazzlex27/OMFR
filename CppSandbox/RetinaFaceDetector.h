#pragma once

#include "Structs.h"
#include <onnxruntime_cxx_api.h>

class RetinaFaceDetector
{
private:
	Ort::Session _session;
	cv::Size _inputSize;
	const int _inputDepth = 3;
	std::map<AnchorKey, Anchor> _anchors;
	const int _featStrideFpn[3] = { 8, 16, 32 };
	const int _numAnchors = 2;

public:
	RetinaFaceDetector(Ort::Env& env, const std::string& modelFilepath);
	std::vector<Face> Detect(const cv::Mat& image, const float detectionThreshold, const float overlapThreshold);

private:
	Anchor CreateAnchor(const AnchorKey& key, const int anchorCount);
	cv::Mat PrepareImage(const cv::Mat& image, float* scaleFactor) const;
	std::vector<std::vector<float>> RunNet(const cv::Mat& floatImage);
	FaceDetectionResult GetResultFromTensorOutput(const std::vector<std::vector<float>>& outputTensorValues, const float threshold,
		const float scaleFactor) const;
	std::vector<Face> ConvertOutput(const FaceDetectionResult& result, const float overlapThreshold, const cv::Size& imageSize) const;
	std::vector<cv::Rect2f> ConvertDistancesToGoodBoxes(const Anchor& anchorCenters, const std::vector<float>& boxPredictions,
		const std::vector<int>& positiveIndexes, const int stride, const float scaleFactor) const;
	std::vector<Landmarks> ConvertDistancesToGoodLms(const Anchor& anchorCenters, const std::vector<float>& lmPredictions,
		const std::vector<int>& positiveIndexes, const int stride, const float scaleFactor) const;
	std::vector<int> ApplyNms(const std::vector<cv::Rect2f>& facesSortedByScore, const float overlapTheshold) const;
};