#include <opencv2/opencv.hpp>

#include "Structs.h"
#include "Utils.h"
#include <numeric>
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

	const int numTests = 100;
	std::vector<long> inferenceTimes;
	inferenceTimes.reserve(numTests);

	auto lastUpdated = std::chrono::steady_clock::now();
	int testsRun = 0;
	const int emulatedFps = 30;
	const float msBetweenFrames = 1000 / emulatedFps;

	while (testsRun < numTests)
	{
		const auto current = std::chrono::steady_clock::now();
		const long lastRunMsElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - lastUpdated).count();
		if (lastRunMsElapsed < msBetweenFrames)
			continue;

		lastUpdated = std::chrono::steady_clock::now();

		const auto begin = std::chrono::steady_clock::now();
		faces = detector.Detect(image, detectionThreshold, overlapThreshold);
		const auto end = std::chrono::steady_clock::now();
		const long msElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
		inferenceTimes.emplace_back(msElapsed);

		testsRun++;
	}

	const long minTime = *std::min_element(inferenceTimes.begin(), inferenceTimes.end());
	const long maxTime = *std::max_element(inferenceTimes.begin(), inferenceTimes.end());
	const long avgTime = std::accumulate(inferenceTimes.begin(), inferenceTimes.end(), 0) / numTests;

	std::cout << "min inference time: " << minTime << " ms" << std::endl;
	std::cout << "max inference time: " << maxTime << " ms" << std::endl;
	std::cout << "avg inference time: "	<< avgTime << " ms" << std::endl;

	cv::Mat copyImage(image);
	DrawFaces(copyImage, faces);
	cv::imwrite("sh_detected.jpg", copyImage);
}