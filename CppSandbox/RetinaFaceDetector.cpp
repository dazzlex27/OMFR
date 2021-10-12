#include "RetinaFaceDetector.h"
#include "OrtUtils.h"
#include <numeric>

RetinaFaceDetector::RetinaFaceDetector(Ort::Env& env, const std::string& modelFilepath)
	:_session(CreateSession(env, modelFilepath))
{
	_inputSize = cv::Size(640, 640);
	for (auto stride : _featStrideFpn)
	{
		const int height = _inputSize.height / stride;
		const int width = _inputSize.width / stride;
		AnchorKey key = { height, width, stride };
		Anchor anchor = CreateAnchor(key, _numAnchors);
		_anchors[key] = anchor;
	}
}

std::vector<Face> RetinaFaceDetector::Detect(const cv::Mat& image, const float detectionThreshold, const float overlapThreshold)
{
	float scaleFactor;
	const cv::Mat& preparedImage = PrepareImage(image, &scaleFactor); // 4-dim float

	const std::vector<std::vector<float>>& outputTensorValues = RunNet(preparedImage);
	const FaceDetectionResult& result = GetResultFromTensorOutput(outputTensorValues, detectionThreshold, scaleFactor);
	const std::vector<Face>& faces = ConvertOutput(result, overlapThreshold);

	return faces;
}

cv::Mat RetinaFaceDetector::PrepareImage(const cv::Mat& image, float* scaleFactor)
{
	const float im_ratio = (float)image.rows / image.cols;
	const float model_ratio = (float)_inputSize.height / _inputSize.width;

	int newWidth = 0;
	int newHeight = 0;

	if (im_ratio > model_ratio)
	{
		newHeight = _inputSize.height;
		newWidth = (int)(newHeight / im_ratio);
		*scaleFactor = (float)newWidth / image.cols;
	}
	else
	{
		newWidth = _inputSize.width;
		newHeight = (int)(newWidth * im_ratio);
		*scaleFactor = (float)newHeight / image.rows;
	}

	cv::Mat resizedImage;
	cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));

	cv::Mat paddedImage = cv::Mat(_inputSize, CV_8UC3);
	cv::Rect roi(cv::Point(0, 0), resizedImage.size());
	resizedImage.copyTo(paddedImage(roi));

	// HWC to CHW
	const float inputStdNorm = 1 / 128.0f;
	const float inputMean = 127.5f;
	const cv::Scalar meanNorm(inputMean, inputMean, inputMean);

	return cv::dnn::blobFromImage(paddedImage, inputStdNorm, _inputSize, meanNorm, true);
}

std::vector<std::vector<float>> RetinaFaceDetector::RunNet(const cv::Mat& floatImage)
{
	Ort::AllocatorWithDefaultOptions allocator;

	// prepare inputs
	size_t numInputNodes = _session.GetInputCount();
	const char* inputName = _session.GetInputName(0, allocator);
	std::vector<const char*> inputNames{ inputName };
	Ort::TypeInfo inputTypeInfo = _session.GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
	std::vector<int64_t> inputDims = { 1, _inputDepth, _inputSize.width, _inputSize.height };
	size_t inputTensorSize = VectorProduct(inputDims);

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	const auto dataPointer = (float*)(floatImage.data);

	std::vector<Ort::Value> inputTensors;
	inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, dataPointer, inputTensorSize, inputDims.data(), inputDims.size()));

	// prepare outputs
	size_t numOutputNodes = _session.GetOutputCount();
	std::vector<const char*> outputNames;
	outputNames.reserve(numOutputNodes);
	std::vector<std::vector<int64_t>> outputDims;
	outputDims.reserve(numOutputNodes);
	std::vector<std::vector<float>> outputTensorValues;
	outputTensorValues.reserve(numOutputNodes);
	std::vector<Ort::Value> outputTensors;
	outputTensors.reserve(numOutputNodes);

	for (int i = 0; i < numOutputNodes; i++)
	{
		const char* outputName = _session.GetOutputName(i, allocator);
		outputNames.emplace_back(outputName);

		Ort::TypeInfo outputTypeInfo = _session.GetOutputTypeInfo(i);
		auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
		outputDims.emplace_back(outputTensorInfo.GetShape());

		size_t outputTensorSize = VectorProduct(outputDims[i]);
		outputTensorValues.emplace_back(std::vector<float>(outputTensorSize)); // reserve space for output values

		outputTensors.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues[i].data(),
			outputTensorValues[i].size(), outputDims[i].data(), outputDims[i].size()));
	}

	// inference
	_session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), numInputNodes,
		outputNames.data(), outputTensors.data(), numOutputNodes);

	return outputTensorValues;
}

FaceDetectionResult RetinaFaceDetector::GetResultFromTensorOutput(const std::vector<std::vector<float>>& outputTensorValues,
	const float threshold, const float scaleFactor)
{
	const int fmc = 3;
	const bool useLandmarks = true;
	const int numAnchors = 2;

	FaceDetectionResult result;

	const int layerCount = sizeof(_featStrideFpn) / sizeof(int);
	for (int i = 0; i < layerCount; i++)
	{
		// parse scores
		const std::vector<float>& scores = outputTensorValues[i];

		std::vector<int> positiveIndexes;
		positiveIndexes.reserve(scores.size());
		for (int j = 0; j < scores.size(); j++)
		{
			if (scores[j] >= threshold)
				positiveIndexes.emplace_back(j);
		}

		result.scores.reserve(scores.size());
		for (int j = 0; j < positiveIndexes.size(); j++)
			result.scores.emplace_back(scores[positiveIndexes[j]]);

		// get anchor
		const int stride = _featStrideFpn[i];
		const int height = _inputSize.height / stride;
		const int width = _inputSize.width / stride;
		AnchorKey key = { height, width, stride };
		Anchor anchor = _anchors[key];

		// parse boxes
		const std::vector<float>& boxPredictions = outputTensorValues[i + fmc];
		const std::vector<cv::Rect2f>& boxes = ConvertDistancesToGoodBoxes(anchor, boxPredictions, positiveIndexes, stride, scaleFactor);
		result.boxes.insert(result.boxes.end(), boxes.begin(), boxes.end());

		if (useLandmarks)
		{
			// parse landmarks
			const std::vector<float>& lmPredictions = outputTensorValues[i + fmc * 2];
			const std::vector<Landmarks>& landmarks = ConvertDistancesToGoodLms(anchor, lmPredictions, positiveIndexes, stride, scaleFactor);
			result.landmarks.insert(result.landmarks.end(), landmarks.begin(), landmarks.end());
		}
	}

	return result;
}

std::vector<Face> RetinaFaceDetector::ConvertOutput(const FaceDetectionResult& result, const float overlapThreshold)
{
	const size_t faceCount = result.boxes.size();

	std::vector<int> indexesSortedByScore = Argsort(result.scores);
	std::reverse(indexesSortedByScore.begin(), indexesSortedByScore.end());

	std::vector<cv::Rect2f> boxesSortedByScore;
	boxesSortedByScore.reserve(faceCount);
	for (int i = 0; i < faceCount; i++)
		boxesSortedByScore.emplace_back(result.boxes[indexesSortedByScore[i]]);

	const size_t lmsCount = result.landmarks.size();

	std::vector<Landmarks> lmsSortedByScore;
	lmsSortedByScore.reserve(lmsCount);
	for (int i = 0; i < lmsCount; i++)
		lmsSortedByScore.emplace_back(result.landmarks[indexesSortedByScore[i]]);

	std::vector<int> validFacesIndexes = ApplyNms(boxesSortedByScore, overlapThreshold);

	const size_t validFaceCount = validFacesIndexes.size();

	std::vector<Face> validFaces;
	validFaces.reserve(validFaceCount);

	for (int i = 0; i < validFaceCount; i++)
	{
		const int index = validFacesIndexes[i];

		Face face;
		face.box = boxesSortedByScore[index];
		face.score = result.scores[index];
		face.landmarks = index < lmsSortedByScore.size() ? lmsSortedByScore[index] : Landmarks();

		validFaces.emplace_back(face);
	}

	return validFaces;
}

std::vector<cv::Rect2f> RetinaFaceDetector::ConvertDistancesToGoodBoxes(const Anchor& anchorCenters,
	const std::vector<float>& boxPredictions, const std::vector<int>& positiveIndexes, const int stride, const float scaleFactor)
{
	const int boxPointCount = 4;
	const size_t positiveIndexCount = positiveIndexes.size();

	std::vector<cv::Rect2f> boxes;
	boxes.reserve(positiveIndexCount);

	for (int i = 0; i < positiveIndexCount; i++)
	{
		const int index = positiveIndexes[i];

		const int boxOffset = boxPointCount * index;
		const std::vector<float>& currentAnchor = anchorCenters[index];

		const float x1 = (currentAnchor[0] - boxPredictions[boxOffset + 0] * stride) / scaleFactor;
		const float y1 = (currentAnchor[1] - boxPredictions[boxOffset + 1] * stride) / scaleFactor;
		const float x2 = (currentAnchor[0] + boxPredictions[boxOffset + 2] * stride) / scaleFactor;
		const float y2 = (currentAnchor[1] + boxPredictions[boxOffset + 3] * stride) / scaleFactor;

		cv::Rect2f box;
		box.x = x1;
		box.y = y1;
		box.width = x2 - x1;
		box.height = y2 - y1;

		boxes.emplace_back(box);
	}

	return boxes;
}

std::vector<Landmarks> RetinaFaceDetector::ConvertDistancesToGoodLms(const Anchor& anchorCenters,
	const std::vector<float>& lmPredictions, const std::vector<int>& positiveIndexes, const int stride, const float scaleFactor)
{
	const int lmPointCount = 5;
	const size_t positiveIndexCount = positiveIndexes.size();

	std::vector<Landmarks> lms;
	lms.reserve(positiveIndexCount);

	for (int i = 0; i < positiveIndexCount; i++)
	{
		const int index = positiveIndexes[i];

		const int lmsOffset = lmPointCount * index;
		const std::vector<float>& currentAnchor = anchorCenters[index];

		std::vector<cv::Point2f> lmSet;
		lmSet.reserve(lmPointCount);

		for (int j = 0; j < lmPointCount; j++)
		{
			const int lmsIndex = (lmsOffset + j) * 2;

			const float x = (currentAnchor[0] + lmPredictions[lmsIndex + 0] * stride) / scaleFactor;
			const float y = (currentAnchor[1] + lmPredictions[lmsIndex + 1] * stride) / scaleFactor;

			lmSet.emplace_back(cv::Point2f(x, y));
		}

		lms.emplace_back(lmSet);
	}

	return lms;
}

std::vector<int> RetinaFaceDetector::ApplyNms(const std::vector<cv::Rect2f>& facesSortedByScore, const float overlapTheshold)
{
	const size_t faceCount = facesSortedByScore.size();
	std::vector<float> areas;
	areas.reserve(faceCount);

	for (int i = 0; i < faceCount; i++)
	{
		const cv::Rect2f& box = facesSortedByScore[i];
		areas.emplace_back((box.width + 1) * (box.height + 1));
	}

	std::vector<int> order(faceCount);
	std::iota(order.begin(), order.end(), 0); // [0 - faceCount-1]

	std::vector<int> validBoxIndexes;
	validBoxIndexes.reserve(faceCount);

	while (order.size() > 0)
	{
		const int index = order[0];
		validBoxIndexes.emplace_back(index);

		const size_t remainingSize = order.size() - 1;

		std::vector<float> overlaps;
		overlaps.reserve(remainingSize);

		const cv::Rect2f& trueBox = facesSortedByScore[index];
		const float trueBoxX2 = trueBox.x + trueBox.width;
		const float trueBoxY2 = trueBox.y + trueBox.height;

		std::vector<int> newOrder;
		newOrder.reserve(remainingSize);

		for (int i = 1; i < order.size(); i++)
		{
			const int currentIndex = order[i];
			const cv::Rect2f& currentBox = facesSortedByScore[currentIndex];
			const float currentBoxX2 = currentBox.x + currentBox.width;
			const float currentBoxY2 = currentBox.y + currentBox.height;

			const float xx1 = std::max(trueBox.x, currentBox.x);
			const float yy1 = std::max(trueBox.y, currentBox.y);
			const float xx2 = std::min(trueBoxX2, currentBoxX2);
			const float yy2 = std::min(trueBoxY2, currentBoxY2);

			const float w = std::max(0.0f, xx2 - xx1 + 1);
			const float h = std::max(0.0f, yy2 - yy1 + 1);
			const float intersectionArea = w * h;

			const float overlap = intersectionArea / (areas[index] + areas[currentIndex] - intersectionArea);
			if (overlap < overlapTheshold)
				newOrder.emplace_back(currentIndex);
		}

		order = newOrder;
	}

	return validBoxIndexes;
}

Anchor RetinaFaceDetector::CreateAnchor(const AnchorKey& key, const int anchorCount)
{
	Anchor anchor;

	const int totalSize = key.width * key.height * anchorCount;

	anchor.reserve(totalSize);

	for (int j = 0; j < key.height; j++)
	{
		const int y = j * key.stride;

		for (int i = 0; i < key.width; i++)
		{
			const int x = i * key.stride;

			for (int k = 0; k < anchorCount; k++)
			{
				std::vector<float> values;
				values.reserve(2);
				values.emplace_back(x);
				values.emplace_back(y);

				anchor.emplace_back(values);
			}
		}
	}

	return anchor;
}