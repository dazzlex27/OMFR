#include "Umeyama.h"

cv::Mat Umeyama::GetSimilarTransform(const cv::Mat& src, const cv::Mat& dst) const
{
	const int num = src.rows;
	const int dim = src.cols;

	const cv::Mat& src_mean = MeanAxis0(src);
	const cv::Mat& dst_mean = MeanAxis0(dst);
	const cv::Mat& src_demean = ElementwiseMinus(src, src_mean);
	const cv::Mat& dst_demean = ElementwiseMinus(dst, dst_mean);
	const cv::Mat& A = (dst_demean.t() * src_demean) / static_cast<float>(num);
	cv::Mat d(dim, 1, CV_32F);
	d.setTo(1.0f);
	if (cv::determinant(A) < 0)
		d.at<float>(dim - 1, 0) = -1;

	const cv::Mat& T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
	cv::Mat U;
	cv::Mat S;
	cv::Mat V;
	cv::SVD::compute(A, S, U, V);

	const int rank = MatrixRank(A);
	if (rank == dim - 1)
	{
		if (cv::determinant(U) * cv::determinant(V) > 0)
			T.rowRange(0, dim).colRange(0, dim) = U * V;
		else
		{
			const int s = d.at<float>(dim - 1, 0) = -1;
			d.at<float>(dim - 1, 0) = -1;

			T.rowRange(0, dim).colRange(0, dim) = U * V;
			const cv::Mat& diag_ = cv::Mat::diag(d);
			const cv::Mat& twp = diag_ * V; //np.dot(np.diag(d), V.T)
			const cv::Mat& B = cv::Mat::zeros(3, 3, CV_8UC1);
			const cv::Mat& C = B.diag(0);
			T.rowRange(0, dim).colRange(0, dim) = U * twp;
			d.at<float>(dim - 1, 0) = s;
		}
	}
	else if (rank != 0)
	{
		const cv::Mat& diag = cv::Mat::diag(d);
		const cv::Mat& twp = diag * V.t(); //np.dot(np.diag(d), V.T)
		const cv::Mat& res = U * twp; // U
		T.rowRange(0, dim).colRange(0, dim) = -U.t() * twp;
	}

	const cv::Mat& var = VarAxis0(src_demean);
	const float val = cv::sum(var).val[0];
	cv::Mat res;
	cv::multiply(d, S, res);
	const float scale = 1.0 / val * cv::sum(res).val[0];
	T.rowRange(0, dim).colRange(0, dim) = -T.rowRange(0, dim).colRange(0, dim).t();
	const cv::Mat& temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
	const cv::Mat& temp2 = src_mean.t(); //src_mean.T
	const cv::Mat& temp3 = temp1 * temp2; // np.dot(T[:dim, :dim], src_mean.T)
	const cv::Mat& temp4 = scale * temp3;
	T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
	T.rowRange(0, dim).colRange(0, dim) *= scale;

	return T;
}

const cv::Mat Umeyama::MeanAxis0(const cv::Mat& src) const
{
	const int num = src.rows;
	const int dim = src.cols;

	// x1 y1
	// x2 y2

	cv::Mat output(1, dim, CV_32F);
	float* op = (float*)output.data;

	for (int i = 0; i < dim; i++)
	{
		float sum = 0;
		for (int j = 0; j < num; j++)
			sum += src.at<float>(j, i);

		op[i] = sum / num;
	}

	return output;
}

const cv::Mat Umeyama::ElementwiseMinus(const cv::Mat& m1, const cv::Mat& m2) const
{
	cv::Mat output(m1.rows, m1.cols, m1.type());

	assert(m2.cols == m1.cols);
	if (m2.cols == m1.cols)
	{
		for (int i = 0; i < m1.rows; i++)
		{
			for (int j = 0; j < m2.cols; j++)
				output.at<float>(i, j) = m1.at<float>(i, j) - m1.at<float>(0, j);
		}
	}

	return output;
}

const cv::Mat Umeyama::VarAxis0(const cv::Mat& src) const
{
	const cv::Mat& mat = ElementwiseMinus(src, MeanAxis0(src));
	cv::multiply(mat, mat, mat);

	return MeanAxis0(mat);
}

const int Umeyama::MatrixRank(const cv::Mat& m) const
{
	cv::Mat w;
	cv::Mat u;
	cv::Mat vt;
	cv::SVD::compute(m, w, u, vt);
	cv::Mat1b nonZeroSingularValues = w > 0.0001;

	return countNonZero(nonZeroSingularValues);
}