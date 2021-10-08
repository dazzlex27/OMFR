using Primitives;
using System;

namespace RecognitionPrimitives.Models
{
	public interface IMaskClassifier : IDisposable
	{
		FaceMaskStatus Classify(ImageData faceImage);
	}
}