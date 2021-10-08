using Primitives;
using RecognitionPrimitives;
using RecognitionPrimitives.Models;

namespace RecognitionEngine
{
	internal class InsightGenderAgeClassifier : IGenderAgeClassifier
	{
		public InsightGenderAgeClassifier(byte[] genderAgeClassifierBytes)
		{

		}

		public IFaceAttributes Classify(ImageData faceImage)
		{
			throw new System.NotImplementedException();
		}

		public void Dispose()
		{
			throw new System.NotImplementedException();
		}
	}
}