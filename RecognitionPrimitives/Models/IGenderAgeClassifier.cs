using Primitives;
using System;

namespace RecognitionPrimitives.Models
{
	public interface IGenderAgeClassifier : IDisposable
	{
		IFaceAttributes Classify(ImageData faceImage);
	}
}