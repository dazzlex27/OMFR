using Primitives;
using Primitives.Logging;
using ProcessingUtils;
using RecognitionEngine;
using System;
using System.IO;
using System.Threading.Tasks;

namespace RecognitionRunner
{
	internal class Program
	{
		static async Task Main(string[] args)
		{
			var logger = new ConsoleLogger();

			try
			{
				var p = new Program();
				await p.Run(logger);
			}
			catch (Exception ex)
			{
				await logger.LogException("Failed to process pipeline", ex);
			}
		}

		private async Task Run(ILogger logger)
		{
			await logger.LogInfo("Starting test...");

			await logger.LogInfo("Loading models...");
			var modelLoader = new ModelFromDiskLoader();
			var modelSet = ModelSetFactory.CreateModels(modelLoader, "models");
			await logger.LogInfo("Loaded models");

			await logger.LogInfo("Creating recognition entities...");
			var faceProcessor = new FaceProcessor(logger, modelSet);
			var indexComparer = FaceIndexComparerFactory.CreateFromIndexType(modelSet.FaceIndexer.IndexType);
			var indexProcessor = new IndexProcessor(logger, indexComparer);
			await logger.LogInfo("Created recognition entities");

			await logger.LogInfo("Loading test images...");
			var imageFolder = "images";
			var image1Path = Path.Combine(imageFolder, "1_0.png");
			var image2Path = Path.Combine(imageFolder, "1_1.png");
			(var image1, var image2) = LoadImages(image1Path, image2Path);
			await logger.LogInfo("Loaded test images");

			await logger.LogInfo("Extracting face data...");
			var image1Faces = faceProcessor.GetFaces(image1);
			var image2Faces = faceProcessor.GetFaces(image2);
			await logger.LogInfo("Extracted face data");

			await logger.LogInfo("Comparing face data...");
			var similarity = indexProcessor.MatchOneToOne(image1Faces[0].FaceIndex, image2Faces[0].FaceIndex);
			await logger.LogInfo($"Similarity between faces = {similarity}");
			await logger.LogInfo("Compared face data");

			if (similarity > 0.7)
				await logger.LogInfo("Face match found!");
			else
				await logger.LogInfo("Match not found");

			await logger.LogInfo("Test finished");
		}

		private static (ImageData image1, ImageData image2) LoadImages(string image1Path, string image2Path)
		{
			var image1 = ImageUtils.ReadImageDataFromFile(image1Path);
			var image2 = ImageUtils.ReadImageDataFromFile(image2Path);

			return (image1, image2);
		}
	}
}