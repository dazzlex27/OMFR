#include <locale>
#include <codecvt>
#include <numeric>
#include <map>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

typedef std::vector<std::vector<float>> Anchor;
typedef std::vector<cv::Point2f> Keypoints;

struct FaceDetectionResult
{
	std::vector<float> scores;
	std::vector<cv::Rect2f> boxes;
	std::vector<Keypoints> keypoints;
};

struct Face
{
	cv::Rect2f box;
	float score;
	Keypoints keypoints;
};

struct AnchorKey
{
	int width;
	int height;
	int stride;
};

bool operator<(const AnchorKey& l, const AnchorKey& r)
{
	return (l.width < r.width || (l.height == r.height && l.stride < r.stride));
}

std::wstring StringToWstring(const std::string& utf8String, const size_t numBytes)
{
	using convert_type = std::codecvt_utf8<typename std::wstring::value_type>;
	std::wstring_convert<convert_type, typename std::wstring::value_type> converter;

	return converter.from_bytes(utf8String.c_str(), utf8String.c_str() + numBytes);
}

template <typename T>
T VectorProduct(const std::vector<T>& v)
{
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::vector<int> Argsort(const std::vector<T>& v)
{
	// initialize original index locations
	std::vector<int> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2]; });

	return idx;
}

cv::Mat PrepareImage(const cv::Mat& image, const cv::Size& inputSize, float* scaleFactor)
{
	const float im_ratio = (float)image.rows / image.cols;
	const float model_ratio = (float)inputSize.height / inputSize.width;

	int newWidth = 0;
	int newHeight = 0;

	if (im_ratio > model_ratio)
	{
		newHeight = inputSize.height;
		newWidth = (int)(newHeight / im_ratio);
	}
	else
	{
		newWidth = inputSize.width;
		newHeight = (int)(newWidth * im_ratio);
	}

	*scaleFactor = (float)newHeight / image.rows;
	cv::Mat resizedImage;
	cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));

	cv::Mat paddedImage = cv::Mat(inputSize, CV_8UC3);
	cv::Rect roi(cv::Point(0, 0), resizedImage.size());
	resizedImage.copyTo(paddedImage(roi));

	return paddedImage;
}

Ort::Session CreateSession(Ort::Env& env, const std::string modelPath)
{
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetIntraOpNumThreads(1);
	bool useCUDA = true;
	if (useCUDA)
	{
		OrtCUDAProviderOptions cuda_options;
		cuda_options.device_id = 0;
		sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
	}

	// Sets graph optimization level
	// Available levels are
	// ORT_DISABLE_ALL -> To disable all optimizations
	// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
	// removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
	// (Includes level 1 + more complex optimizations like node fusions)
	// ORT_ENABLE_ALL -> To Enable All possible optimizations
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	std::wstring modelFilepathW = StringToWstring(modelPath, modelPath.size());

	return Ort::Session(env, modelFilepathW.c_str(), sessionOptions);
}

void PrepareOutputs(const Ort::Session& session, const size_t& numOutputNodes, Ort::AllocatorWithDefaultOptions& allocator,
	Ort::MemoryInfo& memoryInfo, std::vector<std::vector<int64_t>>& outputDims, std::vector<std::vector<float>>& outputTensorValues,
	std::vector<const char*>& outputNames, std::vector<Ort::Value>& outputTensors)
{
	outputDims.reserve(numOutputNodes);
	for (int i = 0; i < numOutputNodes; i++)
		outputDims.emplace_back(std::vector<int64_t>());

	outputTensorValues.reserve(numOutputNodes);
	for (int i = 0; i < numOutputNodes; i++)
		outputTensorValues.emplace_back(std::vector<float>());

	for (int i = 0; i < numOutputNodes; i++)
	{
		const char* outputName = session.GetOutputName(i, allocator);
		outputNames.emplace_back(outputName);

		Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
		auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
		outputDims[i] = outputTensorInfo.GetShape();
		size_t outputTensorSize = VectorProduct(outputDims[i]);
		outputTensorValues[i] = std::vector<float>(outputTensorSize); // reserve space

		outputTensors.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues[i].data(),
			outputTensorValues[i].size(), outputDims[i].data(), outputDims[i].size()));
	}
}

std::vector<cv::Rect2f> ConvertDistancesToBoxes(const Anchor& anchorCenters, const std::vector<float>& boxPredictions)
{
	const int boxPointCount = 4;
	const int boxCount = boxPredictions.size() / boxPointCount;
	assert(boxCount == anchorCenters.size());

	std::vector<cv::Rect2f> boxes;
	boxes.reserve(boxCount);

	for (int i = 0; i < boxCount; i++)
	{
		const int boxOffset = boxPointCount * i;

		const float x1 = anchorCenters[i][0] - boxPredictions[boxOffset + 0];
		const float y1 = anchorCenters[i][1] - boxPredictions[boxOffset + 1];
		const float x2 = anchorCenters[i][0] + boxPredictions[boxOffset + 2];
		const float y2 = anchorCenters[i][1] + boxPredictions[boxOffset + 3];

		cv::Rect2f box;
		box.x = x1;
		box.y = y1;
		box.width = x2 - x1;
		box.height = y2 - y1;

		boxes.emplace_back(box);
	}

	return boxes;
}

std::vector<Keypoints> ConvertDistancesToKeypoints(const Anchor& anchorCenters, const std::vector<float>& keypointsPredictions)
{
	const int keypointsPointCount = 5;
	const int keypointsCount = keypointsPredictions.size() / (keypointsPointCount * 2);
	assert(keypointsCount == anchorCenters.size());

	std::vector<Keypoints> keypoints;
	keypoints.reserve(keypointsCount);

	for (int i = 0; i < keypointsCount; i++)
	{
		const int keypointsOffset = keypointsPointCount * i;
		const std::vector<float> currentAnchor = anchorCenters[i];

		std::vector<cv::Point2f> keypointSet;
		keypointSet.reserve(keypointsPointCount);

		for (int j = 0; j < keypointsPointCount; j++)
		{
			const int keyPointIndex = (keypointsOffset + j) * 2;

			const float x = currentAnchor[0] + keypointsPredictions[keyPointIndex + 0];
			const float y = currentAnchor[1] + keypointsPredictions[keyPointIndex + 1];

			keypointSet.emplace_back(cv::Point2f(x, y));
		}

		keypoints.emplace_back(keypointSet);
	}

	return keypoints;
}

Anchor CreateAnchor(const AnchorKey& key, const int anchorCount)
{
	Anchor anchor;

	const int totalSize = key.width * key.height * anchorCount;

	anchor.reserve(totalSize);

	for (int j = 0; j < key.height; j++)
	{
		for (int i = 0; i < key.width; i++)
		{
			for (int k = 0; k < anchorCount; k++)
			{
				const int x = i * key.stride;
				const int y = j * key.stride;

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

void FillDetectionResultFromTensorOutput(const std::vector<std::vector<float>>& outputTensorValues, const cv::Size& inputSize,
	const float threshold, FaceDetectionResult* result)
{
	const int fmc = 3;
	const bool useKps = true;
	const int numAnchors = 2;
	const int featStrideFpn[] = { 8, 16, 32 };

	std::map<AnchorKey, Anchor> centerCache;

	const int layerCount = sizeof(featStrideFpn) / sizeof(int);

	for (int i = 0; i < layerCount; i++)
	{
		std::vector<float> scores = outputTensorValues[i];
		std::vector<float> rawBoxPredictions = outputTensorValues[i + fmc];
		std::vector<float> boxPredictions;
		boxPredictions.reserve(rawBoxPredictions.size());
		const int stride = featStrideFpn[i];
		for (int j = 0; j < rawBoxPredictions.size(); j++)
			boxPredictions.emplace_back(rawBoxPredictions[j] * stride);

		std::vector<int> positiveIndexes;
		positiveIndexes.reserve(scores.size());
		for (int j = 0; j < scores.size(); j++)
		{
			if (scores[j] >= threshold)
				positiveIndexes.emplace_back(j);
		}

		for (int j = 0; j < positiveIndexes.size(); j++)
			result->scores.emplace_back(scores[positiveIndexes[j]]);

		const int height = inputSize.height / stride;
		const int width = inputSize.width / stride;
		AnchorKey key = { height, width, stride };

		Anchor anchorCenters;

		bool keyExists = centerCache.count(key) == 1;
		if (keyExists)
			anchorCenters = centerCache[key];
		else
		{
			anchorCenters = CreateAnchor(key, numAnchors);
			centerCache[key] = anchorCenters;
		}

		std::vector<cv::Rect2f> boxes = ConvertDistancesToBoxes(anchorCenters, boxPredictions);
		for (int j = 0; j < positiveIndexes.size(); j++)
			result->boxes.emplace_back(boxes[positiveIndexes[j]]);

		if (useKps)
		{
			std::vector<float> keypointsPredictions;
			std::vector<float> rawKeypointsPredictions = outputTensorValues[i + fmc * 2];
			for (int j = 0; j < rawKeypointsPredictions.size(); j++)
				keypointsPredictions.emplace_back(rawKeypointsPredictions[j] * stride);

			std::vector<Keypoints> keypoints = ConvertDistancesToKeypoints(anchorCenters, keypointsPredictions);
			for (int j = 0; j < positiveIndexes.size(); j++)
				result->keypoints.emplace_back(keypoints[positiveIndexes[j]]);
		}
	}
}

void Detect(Ort::Session& session, const cv::Mat& image, const cv::Size& inputSize, const float threshold, FaceDetectionResult* result)
{
	// HWC to CHW
	const float inputMean = 127.5f;
	const float inputStd = 128.0f;
	cv::Mat preprocessedImage = cv::dnn::blobFromImage(image, 1 / inputStd, inputSize, cv::Scalar(inputMean, inputMean, inputMean), true);

	Ort::AllocatorWithDefaultOptions allocator;

	size_t numInputNodes = session.GetInputCount();
	const char* inputName = session.GetInputName(0, allocator);
	std::vector<const char*> inputNames{ inputName };
	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
	std::vector<int64_t> inputDims = { 1, image.channels(), inputSize.width, inputSize.height };
	size_t inputTensorSize = VectorProduct(inputDims);
	std::vector<float> inputTensorValues(inputTensorSize);
	inputTensorValues.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

	std::vector<Ort::Value> inputTensors;
	inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
		inputTensorValues.size(), inputDims.data(), inputDims.size()));

	size_t numOutputNodes = session.GetOutputCount();
	std::vector<std::vector<int64_t>> outputDims;
	std::vector<std::vector<float>> outputTensorValues;
	std::vector<const char*> outputNames;
	std::vector<Ort::Value> outputTensors;
	PrepareOutputs(session, numOutputNodes, allocator, memoryInfo, outputDims, outputTensorValues, outputNames, outputTensors);

	session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), numInputNodes,
		outputNames.data(), outputTensors.data(), numOutputNodes);

	FillDetectionResultFromTensorOutput(outputTensorValues, inputSize, threshold, result);
}

std::vector<int> ApplyNms(const std::vector<cv::Rect2f>& facesSortedByScore, const float overlapTheshold)
{
	const int faceCount = facesSortedByScore.size();
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

		const int remainingSize = order.size() - 1;

		std::vector<float> overlaps;
		overlaps.reserve(remainingSize);

		const cv::Rect2f& trueBox = facesSortedByScore[index];
		const float trueBoxX2 = trueBox.x + trueBox.width;
		const float trueBoxY2 = trueBox.y + trueBox.height;

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
			const float inter = w * h;

			const float overlap = inter / (areas[index] + areas[currentIndex] - inter);
			overlaps.emplace_back(overlap);
		}

		std::vector<int> inds;
		inds.reserve(remainingSize);
		for (int i = 0; i < remainingSize; i++)
		{
			if (overlaps[i] < overlapTheshold)
				inds.emplace_back(i);
		}

		std::vector<int> newOrder;
		newOrder.reserve(order.size());
		for (int i = 0; i < inds.size();i++)
			newOrder.emplace_back(order[inds[i] + 1]);

		order = newOrder;
	}

	return validBoxIndexes;
}

std::vector<Face> ConvertOutput(const FaceDetectionResult& result, const float scaleFactor, const float overlapThreshold)
{
	const int faceCount = result.boxes.size();
	std::vector<cv::Rect2f> correctedBoxes;
	correctedBoxes.reserve(faceCount);

	for (int i = 0; i < faceCount; i++)
	{
		cv::Rect2f box = result.boxes[i];

		const float x2 = box.x + box.width;
		const float y2 = box.y + box.height;

		const float corrX1 = box.x / scaleFactor;
		const float corrY1 = box.y / scaleFactor;
		const float corrX2 = x2 / scaleFactor;
		const float corrY2 = y2 / scaleFactor;

		cv::Rect2f correctedBox(corrX1, corrY1, corrX2 - corrX1, corrY2 - corrY1);
		correctedBoxes.emplace_back(correctedBox);
	}

	const int keypointsCount = result.keypoints.size();
	std::vector<Keypoints> correctedKeypoints;
	correctedKeypoints.reserve(keypointsCount);

	for (int i = 0; i < keypointsCount; i++)
	{
		Keypoints faceKeypoints = result.keypoints[i];

		Keypoints correctedFaceKeypoints;
		correctedKeypoints.reserve(faceKeypoints.size());

		for (int j = 0; j < faceKeypoints.size(); j++)
		{
			const float corrX = faceKeypoints[j].x / scaleFactor;
			const float corrY = faceKeypoints[j].y / scaleFactor;

			correctedFaceKeypoints.emplace_back(cv::Point2f(corrX, corrY));
		}

		correctedKeypoints.emplace_back(correctedFaceKeypoints);
	}

	std::vector<int> indexesSortedByScore = Argsort(result.scores);
	std::reverse(indexesSortedByScore.begin(), indexesSortedByScore.end());

	std::vector<cv::Rect2f> boxesSortedByScore;
	boxesSortedByScore.reserve(faceCount);
	for (int i = 0; i < faceCount; i++)
		boxesSortedByScore.emplace_back(correctedBoxes[indexesSortedByScore[i]]);

	std::vector<Keypoints> keypointsSortedByScore;
	keypointsSortedByScore.reserve(keypointsCount);
	for (int i = 0; i < keypointsCount; i++)
		keypointsSortedByScore.emplace_back(correctedKeypoints[indexesSortedByScore[i]]);

	std::vector<int> validFacesIndexes = ApplyNms(boxesSortedByScore, overlapThreshold);

	const int validFaceCount = validFacesIndexes.size();

	std::vector<Face> validFaces;
	validFaces.reserve(validFaceCount);

	for (int i = 0; i < validFaceCount; i++)
	{
		const int index = validFacesIndexes[i];

		Face face;
		face.box = boxesSortedByScore[index];
		face.score = result.scores[index];
		face.keypoints = index < keypointsSortedByScore.size() ? keypointsSortedByScore[index] : Keypoints();

		validFaces.emplace_back(face);
	}

	return validFaces;
}

void DrawFaces(cv::Mat& image, const std::vector<Face>& faces)
{
	for (Face face : faces)
	{
		cv::rectangle(image, face.box, cv::Scalar(0, 0, 255), 2);
		for (int i = 0; i < face.keypoints.size(); i++)
		{
			cv::Scalar color = i == 0 || i == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
			cv::circle(image, face.keypoints[i], 1, color, 2);
		}
	}
}

int main(int argc, char* argv[])
{
	const OrtLoggingLevel loggingLevel = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
	const char* envName = "retinaface50";
	const char* modelFilepath = "det_10g.onnx";
	cv::Size inputSize(640, 640);
	const char* imageFilepath = "sh.jpg";
	const float detectionThreshold = 0.5f;
	const float overlapThreshold = 0.4f;

	Ort::Env env(loggingLevel, envName);
	Ort::Session session = CreateSession(env, modelFilepath);

	cv::Mat image = cv::imread(imageFilepath);

	float scaleFactor;
	cv::Mat preparedImage = PrepareImage(image, inputSize, &scaleFactor);

	FaceDetectionResult result;
	Detect(session, preparedImage, inputSize, detectionThreshold, &result);

	std::vector<Face> faces = ConvertOutput(result, scaleFactor, overlapThreshold);

	cv::Mat copyImage(image);
	DrawFaces(copyImage, faces);
	cv::imwrite("sh_detected.jpg", copyImage);
}