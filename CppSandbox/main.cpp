#include <locale>
#include <codecvt>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

std::wstring StringToWstring(const std::string& utf8String, const size_t numBytes)
{
	using convert_type = std::codecvt_utf8<typename std::wstring::value_type>;
	std::wstring_convert<convert_type, typename std::wstring::value_type> converter;

	return converter.from_bytes(utf8String.c_str(), utf8String.c_str() + numBytes);
}

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

cv::Mat PrepareImage(const std::string& imageFilepath, const cv::Size& inputSize, float* scaleFactor)
{
	cv::Mat image = cv::imread(imageFilepath);
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
		size_t outputTensorSize = vectorProduct(outputDims[i]);
		outputTensorValues[i] = std::vector<float>(outputTensorSize); // reserve space

		outputTensors.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues[i].data(),
			outputTensorValues[i].size(), outputDims[i].data(), outputDims[i].size()));
	}
}

void Detect(Ort::Session& session, const cv::Mat& image, const cv::Size& inputSize)
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
	size_t inputTensorSize = vectorProduct(inputDims);
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

	const int fmc = 3;
	const bool useKps = true;
	const int numAnchors = 2;
	const float featStrideFpn[] = { 8, 16, 32 };
}

int main(int argc, char* argv[])
{
	Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "inference");
	Ort::Session session = CreateSession(env, "det_10g.onnx");

	cv::Size inputSize(640, 640);

	float scaleFactor;
	cv::Mat preparedImage = PrepareImage("sh.jpg", inputSize, &scaleFactor);

	Detect(session, preparedImage, inputSize);
}