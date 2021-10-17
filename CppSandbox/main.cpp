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
	const cv::Size dstSize(112, 112);
	//float dstMap[lmCount * lmPoints] = { 38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041 };
	float dstMap[lmCount * lmPoints] = { 0.34191, 0.46157, 0.65653, 0.45983, 0.50022, 0.64050, 0.37097, 0.82469, 0.63151, 0.82325 };

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
			absLandmarks.emplace_back(cv::Point2f(x, y));
		}

		int pixelOffsetX = 0;
		int pixelOffsetY = 0;
		float scaleValueX = 0;
		float scaleValueY = 0;
		int maxDim = faceImage.cols;
		if (faceImage.cols < faceImage.rows)
		{
			pixelOffsetX = (faceImage.rows - faceImage.cols) / 2;
			scaleValueX = pixelOffsetX / (float)faceImage.rows;
			maxDim = faceImage.rows;
		}
		if (faceImage.rows < faceImage.cols)
		{
			pixelOffsetY = (faceImage.cols - faceImage.rows) / 2;
			scaleValueY = pixelOffsetY / (float)faceImage.cols;
			maxDim = faceImage.cols;
		}

		cv::Rect absRectPadded(absRect.x - pixelOffsetX, absRect.y - pixelOffsetY, maxDim, maxDim);

		const int pixelOffset = (float)maxDim * 0.3;
		cv::Rect enlargedRect(absRectPadded.x - pixelOffset, absRectPadded.y - pixelOffset,
			absRectPadded.width + pixelOffset*2, absRectPadded.height + pixelOffset*2);
		cv::Mat paddedEnlImage;
		const bool xOk = enlargedRect.x >= 0;
		const bool yOk = enlargedRect.y >= 0;
		const bool wOk = enlargedRect.x + enlargedRect.width < image.cols;
		const bool hOk = enlargedRect.y + enlargedRect.height < image.rows;

		const bool rectInbounds = xOk && yOk && wOk && hOk;
		if (rectInbounds)
			paddedEnlImage = image(enlargedRect);
		else
		{
			const int xOffset = xOk ? 0 : std::abs(enlargedRect.x);
			const int yOffset = yOk ? 0 : std::abs(enlargedRect.y);
			const int wOffset = wOk ? 0 : (enlargedRect.x + enlargedRect.width) - image.cols;
			const int hOffset = hOk ? 0 : (enlargedRect.y + enlargedRect.height) - image.rows;

			cv::Rect enlIntRect(enlargedRect.x + xOffset, enlargedRect.y + yOffset,
				enlargedRect.width - wOffset - xOffset, enlargedRect.height - hOffset - yOffset);

			paddedEnlImage = cv::Mat(cv::Size(enlargedRect.width, enlargedRect.height), CV_8UC3);
			cv::Mat partImage = image(enlIntRect);

			const int reducedWidth = enlargedRect.width - xOffset - wOffset;
			const int reducedHeight = enlargedRect.height - yOffset - hOffset;
			cv::Rect intRect(xOffset, yOffset, reducedWidth, reducedHeight);

			partImage.copyTo(paddedEnlImage(intRect));
		}

		Landmarks correctedLandmarks;
		correctedLandmarks.reserve(face.landmarks.size());

		Landmarks correctedIdLandmarks;
		correctedIdLandmarks.reserve(face.landmarks.size());

		for (int j = 0; j < face.landmarks.size(); j++)
		{
			const float x = face.landmarks[j].x * faceImage.cols / maxDim + scaleValueX;
			const float y = face.landmarks[j].y * faceImage.rows / maxDim + scaleValueY;

			const int corrX = x * maxDim + pixelOffset;
			const int corrY = y * maxDim + pixelOffset;
			cv::Point2f corrLm(corrX, corrY);
			correctedLandmarks.emplace_back(corrLm);

			const int idX = dstMap[j * 2] * maxDim + pixelOffset;
			const int idY = dstMap[j * 2 + 1] * maxDim + pixelOffset;
			cv::Point2f corrId(idX, idY);
			correctedIdLandmarks.emplace_back(corrId);
		}

		cv::Mat srcMat(cv::Size(lmPoints, lmCount), CV_32FC1, (float*)correctedLandmarks.data());
		cv::Mat dstMat(cv::Size(lmPoints, lmCount), CV_32FC1, (float*)correctedIdLandmarks.data());
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

		cv::Mat normImage = cv::Mat(paddedEnlImage.size(), CV_8UC3);
		cv::Mat affine3x2 = cv::Mat(2, 3, CV_32FC1, affine.data);
		cv::warpAffine(paddedEnlImage, normImage, affine3x2, paddedEnlImage.size());

		cv::Rect unpaddedRect(pixelOffset, pixelOffset, normImage.cols - pixelOffset * 2, normImage.rows - pixelOffset * 2);
		cv::Mat unpaddedImage = normImage(unpaddedRect);

		cv::Mat finalImage;
		cv::resize(unpaddedImage, finalImage, dstSize);
		const std::string& finalFaceImagePath = imageFacesFolder + "/" + std::to_string(i) + "_final" + ext;
		cv::imwrite(finalFaceImagePath, finalImage);
	}

	std::cout << std::endl;
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