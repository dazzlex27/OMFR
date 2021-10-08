using RecognitionPrimitives;
using RecognitionPrimitives.Models;

namespace RecognitionEngine
{
	internal class ModelSet : IModelSet
	{
		public ModelSet(IFaceDetector faceDetector, IFaceFilter faceFilter, IFaceLandmarkDetector landmarkDetector,
			IFaceNormalizer faceNormalizer, IFaceIndexer faceIndexer, IGenderAgeClassifier genderAgeClassifier,
			IMaskClassifier maskClassifier)
		{
			FaceDetector = faceDetector;
			FaceFilter = faceFilter;
			LandmarkDetector = landmarkDetector;
			FaceNormalizer = faceNormalizer;
			FaceIndexer = faceIndexer;
			GenderAgeClassifier = genderAgeClassifier;
			MaskClassifier = maskClassifier;
		}

		public IFaceDetector FaceDetector { get; }

		public IFaceFilter FaceFilter { get; }

		public IFaceLandmarkDetector LandmarkDetector { get; }

		public IFaceNormalizer FaceNormalizer { get; }

		public IFaceIndexer FaceIndexer { get; }

		public IGenderAgeClassifier GenderAgeClassifier { get; }

		public IMaskClassifier MaskClassifier { get; }
	}
}