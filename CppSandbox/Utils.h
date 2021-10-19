#pragma once

#include <locale>
#include <codecvt>
#include <filesystem>

namespace fs = std::experimental::filesystem;

class Utils
{
public:
	inline static std::wstring StringToWstring(const std::string& utf8String, const size_t numBytes)
	{
		using convert_type = std::codecvt_utf8<typename std::wstring::value_type>;
		std::wstring_convert<convert_type, typename std::wstring::value_type> converter;

		return converter.from_bytes(utf8String.c_str(), utf8String.c_str() + numBytes);
	}

	template <typename T>
	inline static T VectorProduct(const std::vector<T>& v)
	{
		return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
	}

	template <typename T>
	inline static std::vector<int> Argsort(const std::vector<T>& v)
	{
		// initialize original index locations
		std::vector<int> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0);

		// sort indexes based on comparing values in v
		std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2]; });

		return idx;
	}

	inline static void DrawFaces(cv::Mat& image, const std::vector<Face>& faces)
	{
		for (Face face : faces)
		{
			const int boxAbsX = std::round(face.box.x * image.cols);
			const int boxAbsY = std::round(face.box.y * image.rows);
			const int boxAbsWidth = std::round(face.box.width * image.cols);
			const int boxAbsHeight = std::round(face.box.height * image.rows);

			const cv::Rect absBox(boxAbsX, boxAbsY, boxAbsWidth, boxAbsHeight);
			cv::rectangle(image, absBox, cv::Scalar(0, 0, 255), 2);
			cv::putText(image, face.label, cv::Point(absBox.x, absBox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
			for (int i = 0; i < face.landmarks.size(); i++)
			{
				const int lmAbsX = std::round(boxAbsX + face.landmarks[i].x * boxAbsWidth);
				const int lmAbsY = std::round(boxAbsY + face.landmarks[i].y * boxAbsHeight);

				const cv::Point absPoint(lmAbsX, lmAbsY);

				cv::Scalar color = i == 0 || i == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
				cv::circle(image, absPoint, 1, color, 2);
			}
		}
	}

	inline static void RemoveDirectory(const std::string& directoryName)
	{
		if (fs::exists(directoryName))
			fs::remove_all(directoryName);
	}

	inline static void CreateDirectory(const std::string& directoryName, bool reset = false)
	{
		if (reset)
			RemoveDirectory(directoryName);

		if (!fs::exists(directoryName))
			fs::create_directory(directoryName);
	}
};