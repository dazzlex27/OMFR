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

	NormalizationTest(image, imageFilepath, faces);
	//RetinaFacePerformanceTest(image, detector, detectionThreshold, overlapThreshold, imageFilepath);

	cv::Mat copyImage(image);
	DrawFaces(copyImage, faces);

	fs::path outputPath(imageFilepath);
	const auto fileNameWithoutExt = outputPath.stem();
	const auto fileNameExt = outputPath.extension();

	const std::string& outputFilename = fileNameWithoutExt.string() + "_detected" + fileNameExt.string();
	cv::imwrite(outputFilename, copyImage);
}

void NormalizationTest(const cv::Mat& image, const fs::path& imagePath, const std::vector<Face>& faces)
{
	const std::string& faceFolderName = "faces";

	if (!fs::exists(faceFolderName))
		fs::create_directory(faceFolderName);

	const std::string& imageFacesFolder = faceFolderName + "/" + imagePath.stem().string();
	std::cout << "image faces folder name: " << imageFacesFolder << std::endl;
	if (fs::exists(imageFacesFolder))
	{
		std::cout << "removing existing folder..." << std::endl;
		fs::remove_all(imageFacesFolder);
	}

	fs::create_directory(imageFacesFolder);

	const std::string& ext = imagePath.extension().string();

	const int lmCount = 5;
	const int lmPoints = 2;
	float dstMap[lmCount * lmPoints] = { 38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041 };
	//float dstMapRel[lmCount * lmPoints] =
	//{
	//0.341916071428571,	0.461574107142857,
	//0.656533928571429,	0.459833928571429,
	//0.500225,			0.640505357142857,
	//0.370975892857143,	0.824691964285714,
	//0.631516964285714,	0.823250892857143
	//};
	//cv::Mat dstMat(cv::Size(lmPoints, lmCount), CV_32FC1, dstMap);

	for (int i = 0; i < faces.size(); i++)
	{
		const Face& face = faces[i];

		cv::Rect absRect(face.box.x * image.cols, face.box.y * image.rows, face.box.width * image.cols, face.box.height * image.rows);
		cv::Mat faceImage = image(absRect);

		Landmarks absLandmarks;
		absLandmarks.reserve(face.landmarks.size());
		for (int j = 0; j < face.landmarks.size(); j++)
		{
			const int x = face.landmarks[j].x * absRect.width;
			const int y = face.landmarks[j].y * absRect.height;

			cv::Point2f absPoint(x, y);

			absLandmarks.emplace_back(absPoint);
			cv::circle(faceImage, absPoint, 1, cv::Scalar(0, 255, 0), 2);
		}

		const std::string& faceImagePath = imageFacesFolder + "/" + std::to_string(i) + ext;
		cv::imwrite(faceImagePath, faceImage);

		int pixelOffsetX = 0;
		int pixelOffsetY = 0;
		//float widthScaleValue = 0;
		//float heightScaleValue = 0;
		int maxDim = faceImage.cols;
		if (faceImage.cols < faceImage.rows)
		{
			pixelOffsetX = (faceImage.rows - faceImage.cols) / 2;
			//widthScaleValue = pixelOffset / (float)faceImage.rows;
			maxDim = faceImage.rows;
		}
		if (faceImage.rows < faceImage.cols)
		{
			pixelOffsetY = (faceImage.cols - faceImage.rows) / 2;
			//heightScaleValue = pixelOffset / (float)faceImage.cols;
			maxDim = faceImage.cols;
		}

		cv::Size sqSize(maxDim, maxDim);
		cv::Mat paddedImage = cv::Mat(sqSize, CV_8UC3);
		cv::Rect roi(cv::Point(pixelOffsetX, pixelOffsetY), faceImage.size());
		faceImage.copyTo(paddedImage(roi));

		Landmarks correctedLandmarks;
		correctedLandmarks.reserve(face.landmarks.size());

		Landmarks correctedIdLandmarks;
		correctedIdLandmarks.reserve(face.landmarks.size());

		for (int j = 0; j < face.landmarks.size(); j++)
		{
			const int x = absLandmarks[j].x + pixelOffsetX;
			const int y = absLandmarks[j].y + pixelOffsetY;

			cv::circle(paddedImage, cv::Point(x, y), 1, cv::Scalar(0, 255, 255), 2);
			correctedLandmarks.emplace_back(cv::Point2f(x, y));

			const int idX = dstMap[j * 2] * paddedImage.cols / 112;
			const int idY = dstMap[j * 2 + 1] * paddedImage.rows / 112;
			cv::circle(paddedImage, cv::Point(idX, idY), 1, cv::Scalar(255, 0, 255), 2);
			correctedIdLandmarks.emplace_back(cv::Point2f(idX, idY));
		}

		const std::string& paddedFaceImagePath = imageFacesFolder + "/" + std::to_string(i) + "_padded" + ext;
		cv::imwrite(paddedFaceImagePath, paddedImage);

		cv::Mat srcMat(cv::Size(lmPoints, lmCount), CV_32FC1, (float*)correctedLandmarks.data());
		cv::Mat dstMat(cv::Size(lmPoints, lmCount), CV_32FC1, (float*)correctedIdLandmarks.data());
		//cv::Mat affine = cv::estimateAffine2D(srcMat, dstMat);
		cv::Mat affine = FacePreprocess::similarTransform(srcMat, dstMat);
		for (int j = 0; j < affine.rows; j++)
		{
			for (int i = 0; i < affine.cols; i++)
			{
				float* fp = (float*)affine.data;
				std::cout << fp[j * affine.rows + i] << " ";
			}

			std::cout << std::endl;
		}

		try
		{
			cv::Mat normImage = cv::Mat(paddedImage.size(), CV_8UC3);
			cv::Mat affine3x2 = cv::Mat(2, 3, CV_32FC1, affine.data);
			cv::warpAffine(paddedImage, normImage, affine3x2, paddedImage.size());
			const std::string& normFaceImagePath = imageFacesFolder + "/" + std::to_string(i) + "_norm" + ext;
			cv::imwrite(normFaceImagePath, normImage);
		}
		catch (const cv::Exception& ex)
		{
			std::cout << ex.msg << std::endl;
		}
		catch (const std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
		}
		catch (...) { //everything else
		}

	}
	

	//float srcMap[lmCount * lmPoints] = { 38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041 };
	//float srcMap[lmCount * lmPoints] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };



	std::wcout << std::endl;
}

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