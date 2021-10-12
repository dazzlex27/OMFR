#pragma once

#include <locale>
#include <codecvt>

inline std::wstring StringToWstring(const std::string& utf8String, const size_t numBytes)
{
	using convert_type = std::codecvt_utf8<typename std::wstring::value_type>;
	std::wstring_convert<convert_type, typename std::wstring::value_type> converter;

	return converter.from_bytes(utf8String.c_str(), utf8String.c_str() + numBytes);
}

template <typename T>
inline T VectorProduct(const std::vector<T>& v)
{
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
inline std::vector<int> Argsort(const std::vector<T>& v)
{
	// initialize original index locations
	std::vector<int> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2]; });

	return idx;
}

inline void DrawFaces(cv::Mat& image, const std::vector<Face>& faces)
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