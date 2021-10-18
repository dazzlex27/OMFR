#include "ArcFaceNormalizer.h"
#include "Umeyama.h"

std::vector<cv::Mat> ArcFaceNormalizer::GetNormalizedFaces(const cv::Mat& image, const std::vector<Face>& faces) const
{
	std::vector<cv::Mat> normalizedFaces;
	normalizedFaces.reserve(faces.size());

	for (int i = 0; i < faces.size(); i++)
	{
		const Face& face = faces[i];

		const cv::Rect absRect(face.box.x * image.cols, face.box.y * image.rows, face.box.width * image.cols, face.box.height * image.rows);
		const cv::Mat faceImage = image(absRect);

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

		const cv::Rect absRectPadded(absRect.x - pixelOffsetX, absRect.y - pixelOffsetY, maxDim, maxDim);
		const int pixelOffset = (float)maxDim * 0.3;
		const cv::Rect enlargedRect(absRectPadded.x - pixelOffset, absRectPadded.y - pixelOffset,
			absRectPadded.width + pixelOffset * 2, absRectPadded.height + pixelOffset * 2);
		
		const bool xOk = enlargedRect.x >= 0;
		const bool yOk = enlargedRect.y >= 0;
		const bool wOk = enlargedRect.x + enlargedRect.width < image.cols;
		const bool hOk = enlargedRect.y + enlargedRect.height < image.rows;

		cv::Mat paddedEnlImage;
		const bool rectInbounds = xOk && yOk && wOk && hOk;
		if (rectInbounds)
			paddedEnlImage = image(enlargedRect);
		else
		{
			const int xOffset = xOk ? 0 : std::abs(enlargedRect.x);
			const int yOffset = yOk ? 0 : std::abs(enlargedRect.y);
			const int wOffset = wOk ? 0 : (enlargedRect.x + enlargedRect.width) - image.cols;
			const int hOffset = hOk ? 0 : (enlargedRect.y + enlargedRect.height) - image.rows;
			const int reducedWidth = enlargedRect.width - xOffset - wOffset;
			const int reducedHeight = enlargedRect.height - yOffset - hOffset;

			const cv::Rect enlIntRect(enlargedRect.x + xOffset, enlargedRect.y + yOffset,	reducedWidth, reducedHeight);
			const cv::Rect intRect(xOffset, yOffset, reducedWidth, reducedHeight);

			paddedEnlImage = cv::Mat(cv::Size(enlargedRect.width, enlargedRect.height), CV_8UC3);
			const cv::Mat partImage = image(enlIntRect);
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
			const cv::Point2f corrLm(corrX, corrY);
			correctedLandmarks.emplace_back(corrLm);

			const int idX = _dstMap[j * 2] * maxDim + pixelOffset;
			const int idY = _dstMap[j * 2 + 1] * maxDim + pixelOffset;
			const cv::Point2f corrId(idX, idY);
			correctedIdLandmarks.emplace_back(corrId);
		}

		const cv::Mat srcMat(_lmArraySize, CV_32FC1, (float*)correctedLandmarks.data());
		const cv::Mat dstMat(_lmArraySize, CV_32FC1, (float*)correctedIdLandmarks.data());
		const cv::Mat affine = _transformer.GetSimilarTransform(srcMat, dstMat);
		const cv::Mat affine3x2(2, 3, CV_32FC1, affine.data);

		const cv::Mat normImage(paddedEnlImage.size(), CV_8UC3);
		cv::warpAffine(paddedEnlImage, normImage, affine3x2, paddedEnlImage.size());

		const cv::Rect unpaddedRect(pixelOffset, pixelOffset, normImage.cols - pixelOffset * 2, normImage.rows - pixelOffset * 2);
		const cv::Mat& unpaddedImage = normImage(unpaddedRect);

		normalizedFaces.emplace_back(unpaddedImage);
	}

	return normalizedFaces;
}