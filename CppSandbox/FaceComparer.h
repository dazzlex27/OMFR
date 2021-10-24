#pragma once

#include "Structs.h"

class FaceComparer
{
public:
	const float GetCosineSimilarity(const FaceIndex& index1, const FaceIndex& index2) const;
};