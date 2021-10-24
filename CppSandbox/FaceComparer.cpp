#include "FaceComparer.h"

const float FaceComparer::GetCosineSimilarity(const FaceIndex& index1, const FaceIndex& index2) const
{
	if (index1.size() != index2.size())
		return -1;

	if (index1.size() == 0)
		return -1;

	float* a = (float*)index1.data();
	float* b = (float*)index2.data();

	float dot = 0.0;
	float dotA = 0.0;
	float dotB = 0.0;

	for (int i = 0; i < index1.size(); i++)
	{
		dot += *a * *b;
		dotA += *a * *a;
		dotB += *b * *b;

		a++;
		b++;
	}

	if (dotA == 0 || dotB == 0)
		return -1;

	return dot / (std::sqrt(dotA * dotB));
}