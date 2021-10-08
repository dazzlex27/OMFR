using RecognitionEngine.Models;
using RecognitionPrimitives;
using System.IO;

namespace RecognitionEngine
{
	public static class ModelSetFactory
	{
		public static IModelSet CreateModels(IModelLoader loader, string basePath)
		{
			var faceDetectorBytes = loader.Load(Path.Combine(basePath, "fd", "retina50.onnx"));
			var faceDetector = new Retina50FaceDetector(faceDetectorBytes);

			var faceFilterBytes = loader.Load(Path.Combine(basePath, "fi", "filter1.onnx"));
			var faceFilter = new ConvNetFaceFilter(faceFilterBytes);

			var landmarkDetectorBytes = loader.Load(Path.Combine(basePath, "fl", "insight_68_landmarks.onnx"));
			var landmarkDetector = new InsightFace68LandmarkDetector(landmarkDetectorBytes);

			var faceNormalizer = new InsightFaceNormalizer();

			var faceIndexerBytes = loader.Load(Path.Combine(basePath, "fi", "arcface50.onnx"));
			var faceIndexer = new ArcFace50FaceIndexer(faceIndexerBytes);

			var genderAgeClassifierBytes = loader.Load(Path.Combine(basePath, "gac", "insight_gender_age.onnx"));
			var genderAgeClassifier = new InsightGenderAgeClassifier(genderAgeClassifierBytes);

			var maskClassifierBytes = loader.Load(Path.Combine(basePath, "mc", "mask1.onnx"));
			var maskClassifier = new ConvNetMaskClassifier(maskClassifierBytes);

			return new ModelSet(faceDetector, faceFilter, landmarkDetector, faceNormalizer, faceIndexer,
				genderAgeClassifier, maskClassifier);
		}
	}
}