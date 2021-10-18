#include <numeric>
#include <filesystem>
#include "Structs.h"
#include "Utils.h"
#include "RetinaFaceDetector.h"
#include "ArcFaceNormalizer.h"
#include "ArcFace50Indexer.h"
#include <fstream>

namespace fs = std::experimental::filesystem;

void RetinaFacePerformanceTest(const cv::Mat& image, RetinaFaceDetector& detector, const float detectionThreshold,
	const float overlapThreshold);
void NormalizationPerformanceTest(const cv::Mat& image, const ArcFaceNormalizer& normalizer, const std::vector<Face>& faces);
void IndexingPerformanceTest(ArcFace50Indexer& indexer, const std::vector<cv::Mat>& faceImages);
void SaveDetectionResult(const cv::Mat& image, const std::vector<Face>& faces,
	const fs::path& imagePath, const std::string& imageFacesFolder);
void SaveNormalizationResult(const std::vector<cv::Mat>& normalizedFaces, const fs::path& imagePath,
	const std::string& imageFacesFolder);
void SaveIndexingResult(const std::vector<Face>& faces, const std::string& imageFacesFolder);

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
	std::cout << std::endl;
#else
	std::string imageFilepath = "sh.jpg";
#endif

	const char* detectorModelFilepath = "det_10g.onnx";
	const char* indexerModelFilepath = "w600k_r50.onnx";
	const float detectionThreshold = 0.5f;
	const float overlapThreshold = 0.4f;

	cv::Mat image = cv::imread(imageFilepath);

	Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "inference");

	RetinaFaceDetector detector(env, detectorModelFilepath);
	std::vector<Face> faces = detector.Detect(image, detectionThreshold, overlapThreshold);

	ArcFaceNormalizer normalizer;
	const std::vector<cv::Mat>& normalizedFaces = normalizer.GetNormalizedFaces(image, faces);

	const cv::Size dstSize(112, 112);
	std::vector<cv::Mat> scaledNormImages;
	scaledNormImages.reserve(normalizedFaces.size());
	for (int i = 0; i < normalizedFaces.size(); i++)
	{
		cv::Mat scaledNormImage;
		cv::resize(normalizedFaces[i], scaledNormImage, dstSize);
		scaledNormImages.emplace_back(scaledNormImage);
	}

	ArcFace50Indexer indexer(env, indexerModelFilepath);
	for (int i = 0; i < scaledNormImages.size(); i++)
		faces[i].index = indexer.GetIndex(scaledNormImages[i]);

	RetinaFacePerformanceTest(image, detector, detectionThreshold, overlapThreshold);
	NormalizationPerformanceTest(image, normalizer, faces);
	IndexingPerformanceTest(indexer, scaledNormImages);

	fs::path imagePath(imageFilepath);
	const std::string& faceFolderName = "faces";
	CreateDirectory(faceFolderName);
	const std::string& imageFacesFolder = faceFolderName + "/" + imagePath.stem().string();
	CreateDirectory(imageFacesFolder, true);

	SaveDetectionResult(image, faces, imagePath, imageFacesFolder);
	SaveNormalizationResult(normalizedFaces, imagePath, imageFacesFolder);
	SaveIndexingResult(faces, imageFacesFolder);
}

void RetinaFacePerformanceTest(const cv::Mat& image, RetinaFaceDetector& detector, const float detectionThreshold,
	const float overlapThreshold)
{
	std::cout << "starting RetinaFace performance test..." << std::endl;

	const int numTests = 100;
	std::vector<int> runTimes;
	runTimes.reserve(numTests);

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
		runTimes.emplace_back((int)msElapsed);

		testsRun++;
	}

	const long minTime = *std::min_element(runTimes.begin(), runTimes.end());
	const long maxTime = *std::max_element(runTimes.begin(), runTimes.end());
	const float avgTime = std::accumulate(runTimes.begin(), runTimes.end(), 0) / (float)numTests;
	const float potentialFps = 1000.0f / avgTime;

	std::cout << "finished RetinaFace performance test:" << std::endl;
	std::cout << "image size: " << image.cols << "x" << image.rows << std::endl;
	std::cout << "face count: " << faces.size() << std::endl;
	std::cout << "min detection time: " << minTime << " ms" << std::endl;
	std::cout << "max detection time: " << maxTime << " ms" << std::endl;
	std::cout << "avg detection time: " << avgTime << " ms" << " (fps=" << potentialFps << ")" << std::endl;
	std::cout << std::endl;
}

void NormalizationPerformanceTest(const cv::Mat& image, const ArcFaceNormalizer& normalizer, const std::vector<Face>& faces)
{
	std::cout << "starting normalization performance test..." << std::endl;

	const int numTests = 100;
	std::vector<int> runTimes;
	runTimes.reserve(numTests);

	auto lastUpdated = std::chrono::steady_clock::now();
	int testsRun = 0;
	const int emulatedFps = 30;
	const float msBetweenFrames = 1000 / emulatedFps;

	std::vector<cv::Mat> normalizedFaces;
	normalizedFaces.reserve(500);

	while (testsRun < numTests)
	{
		const auto current = std::chrono::steady_clock::now();
		const auto lastRunMsElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - lastUpdated).count();
		if (lastRunMsElapsed < msBetweenFrames)
			continue;

		lastUpdated = std::chrono::steady_clock::now();

		const auto begin = std::chrono::steady_clock::now();
		normalizedFaces = normalizer.GetNormalizedFaces(image, faces);
		const auto end = std::chrono::steady_clock::now();
		const auto msElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
		runTimes.emplace_back((int)msElapsed);

		testsRun++;
	}

	const long minTime = *std::min_element(runTimes.begin(), runTimes.end());
	const long maxTime = *std::max_element(runTimes.begin(), runTimes.end());
	const float avgTime = std::accumulate(runTimes.begin(), runTimes.end(), 0) / (float)numTests;
	const float potentialFps = 1000.0f / avgTime;

	std::cout << "finished normalization performance test:" << std::endl;
	std::cout << "image size: " << image.cols << "x" << image.rows << std::endl;
	std::cout << "face count: " << normalizedFaces.size() << std::endl;
	std::cout << "min normalization time: " << minTime << " ms" << std::endl;
	std::cout << "max normalization time: " << maxTime << " ms" << std::endl;
	std::cout << "avg normalization time: " << avgTime << " ms" << " (fps=" << potentialFps << ")" << std::endl;
	std::cout << std::endl;
}

void IndexingPerformanceTest(ArcFace50Indexer& indexer, const std::vector<cv::Mat>& faceImages)
{
	std::cout << "starting indexing performance test..." << std::endl;

	const int numTests = 100;
	std::vector<int> runTimes;
	runTimes.reserve(numTests);

	auto lastUpdated = std::chrono::steady_clock::now();
	int testsRun = 0;
	const int emulatedFps = 30;
	const float msBetweenFrames = 1000 / emulatedFps;

	FaceIndex index;

	int i = 0;

	while (testsRun < numTests)
	{
		const auto current = std::chrono::steady_clock::now();
		const auto lastRunMsElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - lastUpdated).count();
		if (lastRunMsElapsed < msBetweenFrames)
			continue;

		lastUpdated = std::chrono::steady_clock::now();

		if (i == faceImages.size())
			i = 0;

		const auto begin = std::chrono::steady_clock::now();
		index = indexer.GetIndex(faceImages[i]);
		const auto end = std::chrono::steady_clock::now();
		const auto msElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
		runTimes.emplace_back((int)msElapsed);

		testsRun++;
		i++;
	}

	const long minTime = *std::min_element(runTimes.begin(), runTimes.end());
	const long maxTime = *std::max_element(runTimes.begin(), runTimes.end());
	const float avgTime = std::accumulate(runTimes.begin(), runTimes.end(), 0) / (float)numTests;
	const float potentialFps = 1000.0f / avgTime;

	std::cout << "finished indexing performance test:" << std::endl;
	std::cout << "face count: " << faceImages.size() << std::endl;
	std::cout << "min indexing time: " << minTime << " ms" << std::endl;
	std::cout << "max indexing time: " << maxTime << " ms" << std::endl;
	std::cout << "avg indexing time: " << avgTime << " ms" << " (fps=" << potentialFps << ")" << std::endl;
	std::cout << std::endl;
}

void SaveDetectionResult(const cv::Mat& image, const std::vector<Face>& faces,
	const fs::path& imagePath, const std::string& imageFacesFolder)
{
	cv::Mat copyImage = image.clone();
	DrawFaces(copyImage, faces);

	fs::path outputPath(imagePath);
	const auto fileNameWithoutExt = outputPath.stem();
	const auto fileNameExt = outputPath.extension();

	const std::string& outputFilename = imageFacesFolder + "/" + "detected" + fileNameExt.string();
	cv::imwrite(outputFilename, copyImage);
}

void SaveNormalizationResult(const std::vector<cv::Mat>& normalizedFaces, const fs::path& imagePath,
	const std::string& imageFacesFolder)
{
	const std::string normFacesFolderName(imageFacesFolder + "/" + "normalized");
	CreateDirectory(normFacesFolderName);
	const std::string& ext = imagePath.extension().string();

	for (int i = 0; i < normalizedFaces.size(); i++)
	{
		const cv::Size dstSize(112, 112);
		cv::Mat finalImage;
		cv::resize(normalizedFaces[i], finalImage, dstSize);
		const std::string& finalFaceImagePath = normFacesFolderName + "/" + std::to_string(i) + "_norm" + ext;
		cv::imwrite(finalFaceImagePath, finalImage);
	}
}

void SaveIndexingResult(const std::vector<Face>& faces, const std::string& imageFacesFolder)
{
	const std::string indexFolderName(imageFacesFolder + "/" + "indexes");
	CreateDirectory(indexFolderName);

	for (int i = 0; i < faces.size(); i++)
	{
		const std::string& indexFilepath = indexFolderName + "/" + std::to_string(i) + ".txt";
		std::ofstream file;
		file.open(indexFilepath);
		for (int j = 0; j < faces[i].index.size(); j++)
			file << faces[i].index[j] << " ";
		file << std::endl;
		file.close();
	}
}