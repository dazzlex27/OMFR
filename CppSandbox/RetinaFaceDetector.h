#pragma once

#include "Structs.h"
#include <onnxruntime_cxx_api.h>

class RetinaFaceDetector
{
private:
	Ort::Session _session;
	cv::Size _inputSize;

public:
	RetinaFaceDetector(Ort::Env& env, const std::string& modelFilepath);
	std::vector<Face> Detect(const cv::Mat& image, const float detectionThreshold, const float overlapThreshold);


private:
	cv::Mat PrepareImage(const cv::Mat& image, float* scaleFactor);
	void PrepareOutputs(const size_t& numOutputNodes, Ort::AllocatorWithDefaultOptions& allocator,
		Ort::MemoryInfo& memoryInfo, std::vector<std::vector<int64_t>>& outputDims, std::vector<std::vector<float>>& outputTensorValues,
		std::vector<const char*>& outputNames, std::vector<Ort::Value>& outputTensors);
	std::vector<cv::Rect2f> ConvertDistancesToBoxes(const Anchor& anchorCenters, const std::vector<float>& boxPredictions);
	std::vector<Keypoints> ConvertDistancesToKeypoints(const Anchor& anchorCenters, const std::vector<float>& keypointsPredictions);
	Anchor CreateAnchor(const AnchorKey& key, const int anchorCount);
	void FillDetectionResultFromTensorOutput(const std::vector<std::vector<float>>& outputTensorValues,
		const float threshold, FaceDetectionResult* result);
	void RunNet(const cv::Mat& image, const float threshold, FaceDetectionResult* result);
	std::vector<int> ApplyNms(const std::vector<cv::Rect2f>& facesSortedByScore, const float overlapTheshold);
	std::vector<Face> ConvertOutput(const FaceDetectionResult& result, const float scaleFactor, const float overlapThreshold);
};