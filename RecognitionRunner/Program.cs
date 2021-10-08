using Primitives;
using ProcessingUtils;
using RecognitionEngine;
using System;
using System.IO;

namespace RecognitionRunner
{
	internal class Program
	{
		static void Main(string[] args)
		{
			var p = new Program();
			p.Run();
		}

		private void Run()
		{
			Console.WriteLine("Starting test...");

			Console.WriteLine("Loading models...");
			var modelLoader = new ModelLoader();
			var modelSet = modelLoader.LoadModelsFromDisk("models");
			var faceProcessor = new FaceProcessor(modelSet);
			var indexProcessor = new IndexProcessor();
			Console.WriteLine("Loaded models");

			Console.WriteLine("Loading images...");

			var imageFolder = "images";

			var image1Path = Path.Combine(imageFolder, "1_0.png");
			var image2Path = Path.Combine(imageFolder, "1_1.png");

			(var image1, var image2) = LoadImages(image1Path, image2Path);

			Console.WriteLine("Loaded images");

			Console.WriteLine("Extracting face data...");

			var image1Faces = faceProcessor.GetFaces(image1);
			var image2Faces = faceProcessor.GetFaces(image2);

			Console.WriteLine("Extracted face data");

			Console.WriteLine("Comparing face data...");

			var similarity = indexProcessor.MatchOneToOne(image1Faces[0].FaceIndex, image2Faces[0].FaceIndex);
			Console.WriteLine($"Similarity between faces = {similarity}");

			Console.WriteLine("Compared face data");

			if (similarity > 0.7)
				Console.WriteLine("Face match found!");
			else
				Console.WriteLine("Match not found");

			Console.WriteLine("Test finished");
		}

		private static (ImageData image1, ImageData image2) LoadImages(string image1Path, string image2Path)
		{
			var image1 = ImageUtils.ReadImageDataFromFile(image1Path);
			var image2 = ImageUtils.ReadImageDataFromFile(image2Path);

			return (image1, image2);
		}
	}
}