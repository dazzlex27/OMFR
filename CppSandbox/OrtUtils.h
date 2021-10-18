#pragma once

#include <onnxruntime_cxx_api.h>
#include "Utils.h"

inline Ort::Session CreateSession(Ort::Env& env, const std::string modelPath)
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
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	std::wstring modelFilepathW = StringToWstring(modelPath, modelPath.size());

	return Ort::Session(env, modelFilepathW.c_str(), sessionOptions);
}