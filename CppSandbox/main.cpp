#include <numeric>
#include <filesystem>
#include "Structs.h"
#include "Utils.h"
#include "RetinaFaceDetector.h"
#include "Umeyama.h"

namespace fs = std::experimental::filesystem;

void NormalizationTest(const cv::Mat& image, const fs::path& imagePath, const std::vector<Face>& faces);
void RetinaFacePerformanceTest(const cv::Mat& image, RetinaFaceDetector& detector, const float detectionThreshold,
	const float overlapThreshold, const std::string& inputFilename);

int main(int argc, char* argv[])
{
#ifdef NDEBUG
	if (argc < 2)
	{
		std::cout << "image name not provided" << std::endl;
		return -1;
	}

	const std::string& imageFilepath = argv[1];
	std::cout << "image name: " << imageFilepath << std::endl;
#else
	std::string imageFilepath = "sh.jpg";
#endif

	const char* modelFilepath = "det_10g.onnx";
	const float detectionThreshold = 0.5f;
	const float overlapThreshold = 0.4f;

	cv::Mat image = cv::imread(imageFilepath);

	Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "inference");

	RetinaFaceDetector detector(env, modelFilepath);

	std::vector<Face> faces = detector.Detect(image, detectionThreshold, overlapThreshold);

	//NormalizationTest(image, imageFilepath, faces);
	//RetinaFacePerformanceTest(image, detector, detectionThreshold, overlapThreshold, imageFilepath);

	cv::Mat copyImage(image);
	DrawFaces(copyImage, faces);

	fs::path outputPath(imageFilepath);
	const auto fileNameWithoutExt = outputPath.stem();
	const auto fileNameExt = outputPath.extension();

	const std::string& outputFilename = fileNameWithoutExt.string() + "_detected" + fileNameExt.string();
	cv::imwrite(outputFilename, copyImage);
}

//void NormalizationTest(const cv::Mat& image, const fs::path& imagePath, const std::vector<Face>& faces)
//{
//	const std::string& faceFolderName = "faces";
//
//	if (!fs::exists(faceFolderName))
//		fs::create_directory(faceFolderName);
//
//	const std::string& imageFacesFolder = faceFolderName + "/" + imagePath.stem().string();
//	std::cout << "image faces folder name: " << imageFacesFolder << std::endl;
//	if (fs::exists(imageFacesFolder))
//	{
//		std::cout << "removing existing folder..." << std::endl;
//		fs::remove_all(imageFacesFolder);
//	}
//
//	fs::create_directory(imageFacesFolder);
//
//	const int lmCount = 5;
//	const int lmPoints = 2;
//	float dstMap[lmCount * lmPoints] = { 38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041 };
//	/*float dstMapRel[lmCount * lmPoints] =
//	{
//	0.341916071428571,	0.461574107142857,
//	0.656533928571429,	0.459833928571429,
//	0.500225,			0.640505357142857,
//	0.370975892857143,	0.824691964285714,
//	0.631516964285714,	0.823250892857143
//	};*/
//	cv::Mat dstMat(cv::Size(lmPoints, lmCount), CV_32FC1, dstMap);
//
//	for (int i = 0; i < faces.size(); i++)
//	{
//		cv::Rect abs
//
//		cv::Mat roi()
//	}
//	
//
//	float srcMap[lmCount * lmPoints] = { 38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041 };
//	//float srcMap[lmCount * lmPoints] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
//
//	cv::Mat srcMat(cv::Size(lmPoints, lmCount), CV_32FC1, srcMap);
//
//	cv::Mat affine = FacePreprocess::similarTransform(srcMat, dstMat);
//	for (int j=0;j<affine.rows;j++)
//	{ 
//		for (int i = 0; i < affine.cols; i++)
//		{
//			float* fp = (float*)affine.data;
//			std::cout << fp[j * affine.rows + i] << " ";
//		}
//
//		std::cout << std::endl;
//	}
//
//	std::wcout << std::endl;
//}

void RetinaFacePerformanceTest(const cv::Mat& image, RetinaFaceDetector& detector, const float detectionThreshold,
	const float overlapThreshold, const std::string& inputFilename)
{
	const int numTests = 100;
	std::vector<int> inferenceTimes;
	inferenceTimes.reserve(numTests);

	auto lastUpdated = std::chrono::steady_clock::now();
	int testsRun = 0;
	const int emulatedFps = 30;
	const float msBetweenFrames = 1000 / emulatedFps;

	std::vector<Face> faces;
	faces.reserve(500);

	while (testsRun < numTests)
	{
		const auto current = std::chrono::steady_clock::now();
		const auto lastRunMsElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - lastUpdated).count();
		if (lastRunMsElapsed < msBetweenFrames)
			continue;

		lastUpdated = std::chrono::steady_clock::now();

		const auto begin = std::chrono::steady_clock::now();
		faces = detector.Detect(image, detectionThreshold, overlapThreshold);
		const auto end = std::chrono::steady_clock::now();
		const auto msElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
		inferenceTimes.emplace_back((int)msElapsed);

		testsRun++;
	}

	const long minTime = *std::min_element(inferenceTimes.begin(), inferenceTimes.end());
	const long maxTime = *std::max_element(inferenceTimes.begin(), inferenceTimes.end());
	const float avgTime = std::accumulate(inferenceTimes.begin(), inferenceTimes.end(), 0) / (float)numTests;
	const float potentialFps = 1000.0f / avgTime;

	std::cout << "face count: " << faces.size() << std::endl;
	std::cout << "min inference time: " << minTime << " ms" << std::endl;
	std::cout << "max inference time: " << maxTime << " ms" << std::endl;
	std::cout << "avg inference time: " << avgTime << " ms" << " (fps=" << potentialFps << ")" << std::endl;
}