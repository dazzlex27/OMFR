using Primitives;
using RecognitionPrimitives;
using RecognitionPrimitives.Models;

namespace RecognitionEngine
{
	internal class ConvNetMaskClassifier : IMaskClassifier
	{
		public ConvNetMaskClassifier(byte[] maskClassifierBytes)
		{

		}

		public FaceMaskStatus Classify(ImageData faceImage)
		{
			throw new System.NotImplementedException();
		}

		public void Dispose()
		{
			throw new System.NotImplementedException();
		}
	}
}