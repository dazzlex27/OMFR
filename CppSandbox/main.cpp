#include <opencv2/opencv.hpp>

#include "Structs.h"
#include "Utils.h"
#include "RetinaFaceDetector.h"

int main(int argc, char* argv[])
{
	const char* modelFilepath = "det_10g.onnx";
	const char* imageFilepath = "sh.jpg";
	const float detectionThreshold = 0.5f;
	const float overlapThreshold = 0.4f;

	cv::Mat image = cv::imread(imageFilepath);

	Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "inference");

	RetinaFaceDetector detector(env, modelFilepath);

	std::vector<Face> faces = detector.Detect(image, detectionThreshold, overlapThreshold);

	cv::Mat copyImage(image);
	DrawFaces(copyImage, faces);
	cv::imwrite("sh_detected.jpg", copyImage);
}