using Primitives;
using Primitives.Logging;
using RecognitionPrimitives;
using RecognitionPrimitives.Models;
using System.Collections.Generic;

namespace RecognitionRunner
{
	public class FaceProcessor
	{
		private readonly ILogger _logger;

		private readonly IFaceDetector _faceDetector;
		private readonly IFaceFilter _faceFilter;
		private readonly IFaceLandmarkDetector _landmarkDetector;
		private readonly IFaceNormalizer _faceNormalizer;
		private readonly IGenderAgeClassifier _genderAgeClassifier;
		private readonly IMaskClassifier _maskClassifier;
		private readonly IFaceIndexer _faceIndexer;

		public FaceProcessor(ILogger logger, IModelSet modelSet)
		{
			_logger = logger;
			_logger.LogInfo("Creating face processor...");

			_faceDetector = modelSet.FaceDetector;
			_faceFilter = modelSet.FaceFilter;
			_landmarkDetector = modelSet.LandmarkDetector;
			_faceNormalizer = modelSet.FaceNormalizer;
			_genderAgeClassifier = modelSet.GenderAgeClassifier;
			_maskClassifier = modelSet.MaskClassifier;
			_faceIndexer = modelSet.FaceIndexer;
		}

		public IReadOnlyList<IFaceInfo> GetFaces(ImageData image)
		{
			var detectedFaces = _faceDetector.Detect(image);
			var filteredFaces = _faceFilter.GetFilteredFaces(image, detectedFaces);
			var facesLandmarks = _landmarkDetector.GetFacesLandmarks(filteredFaces);
			var normalizedFaces = _faceNormalizer.Normalize(filteredFaces, facesLandmarks);

			var result = new List<IFaceInfo>();
			foreach(var faceImage in normalizedFaces)
			{
				var faceIndex = _faceIndexer.GetFaceIndex(faceImage);
				var maskStatus = _maskClassifier.Classify(faceImage);
				var faceAttributes = _genderAgeClassifier.Classify(faceImage);

				var faceInfo = new FaceInfo(faceImage, faceIndex, maskStatus, faceAttributes.Gender, faceAttributes.Age);
				result.Add(faceInfo);
			}

			return result;
		}

		public void Dispose()
		{
			_faceDetector.Dispose();
			_faceFilter.Dispose();
			_landmarkDetector.Dispose();
			_faceNormalizer.Dispose();
			_genderAgeClassifier.Dispose();
			_maskClassifier.Dispose();
			_faceIndexer.Dispose();
		}
	}
}