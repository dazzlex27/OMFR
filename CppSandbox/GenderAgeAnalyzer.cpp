#include "GenderAgeAnalyzer.h"
#include "OrtUtils.h"
#include <numeric>

GenderAgeAnalyzer::GenderAgeAnalyzer(Ort::Env& env, const std::string& modelFilepath)
	:_session(CreateSession(env, modelFilepath))
{
}

GenderAgeAttributes GenderAgeAnalyzer::GetAttributes(const cv::Mat& faceImage)
{
	float scaleFactor = 1;
	const cv::Mat& preparedImage = PrepareImage(faceImage, &scaleFactor); // 4-dim float

	const std::vector<float>& tensorOutput = RunNet(preparedImage);

	return GetResultFromTensorOutput(tensorOutput);
}

cv::Mat GenderAgeAnalyzer::PrepareImage(const cv::Mat& image, float* scaleFactor) const
{
	cv::Mat paddedImage;

	const bool imgSizeMatches = image.cols == _inputSize.width && image.rows == _inputSize.height;
	if (imgSizeMatches)
		paddedImage = image;
	else
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

		paddedImage = cv::Mat(_inputSize, CV_8UC3);
		cv::Rect roi(cv::Point(0, 0), resizedImage.size());
		resizedImage.copyTo(paddedImage(roi));
	}

	// HWC to CHW
	const float inputStdNorm = 1 / 128.0f;
	const float inputMean = 127.5f;
	const cv::Scalar meanNorm(inputMean, inputMean, inputMean);

	return cv::dnn::blobFromImage(paddedImage, inputStdNorm, _inputSize, meanNorm, true);
}

std::vector<float> GenderAgeAnalyzer::RunNet(const cv::Mat& floatImage)
{
	Ort::AllocatorWithDefaultOptions allocator;

	// prepare inputs
	const char* inputName = _session.GetInputName(0, allocator);
	std::vector<const char*> inputNames{ inputName };
	Ort::TypeInfo inputTypeInfo = _session.GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
	std::vector<int64_t> inputDims = { 1, _inputDepth, _inputSize.width, _inputSize.height };
	size_t inputTensorSize = Utils::VectorProduct(inputDims);

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	const auto dataPointer = (float*)(floatImage.data);

	std::vector<Ort::Value> inputTensors;
	inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, dataPointer, inputTensorSize, inputDims.data(), inputDims.size()));

	// prepare outputs
	const char* outputName = _session.GetOutputName(0, allocator);
	std::vector<const char*> outputNames{ outputName };
	Ort::TypeInfo outputTypeInfo = _session.GetOutputTypeInfo(0);
	auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
	const auto outputDims = outputTensorInfo.GetShape();
	size_t outputTensorSize = Utils::VectorProduct(outputDims);

	std::vector<float> outputTensorValue(outputTensorSize); // reserve space for output values

	std::vector<Ort::Value> outputTensors;
	outputTensors.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValue.data(),
		outputTensorValue.size(), outputDims.data(), outputDims.size()));

	// inference
	_session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), 1,
		outputNames.data(), outputTensors.data(), 1);

	return outputTensorValue;
}

GenderAgeAttributes GenderAgeAnalyzer::GetResultFromTensorOutput(const std::vector<float>& tensorOutput) const
{
	GenderAgeAttributes attributes;

	if (tensorOutput.size() != 3)
		return attributes;

	bool isMale = false;

	if (tensorOutput[0] > tensorOutput[1])
		isMale = true;

	attributes.first = isMale ? Gender::Male : Gender::Female;
	attributes.second = (int)(std::round(tensorOutput[2] * 100));

	return attributes;
}