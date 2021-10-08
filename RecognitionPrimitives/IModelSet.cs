using RecognitionPrimitives.Models;

namespace RecognitionPrimitives
{
	public interface IModelSet
	{
		IFaceDetector FaceDetector { get; }

		IFaceFilter FaceFilter { get; }

		IFaceLandmarkDetector LandmarkDetector { get; }

		IFaceNormalizer FaceNormalizer { get; }

		IFaceIndexer FaceIndexer { get; }

		IGenderAgeClassifier GenderAgeClassifier { get; }

		IMaskClassifier MaskClassifier { get; }
	}
}