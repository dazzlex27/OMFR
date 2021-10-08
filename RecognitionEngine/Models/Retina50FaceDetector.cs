using Primitives;
using Primitives.Structs;
using RecognitionPrimitives.Models;
using System.Collections.Generic;

namespace RecognitionEngine.Models
{
	internal class Retina50FaceDetector : IFaceDetector
	{
		public Retina50FaceDetector(byte[] modelBytes)
		{

		}

		public IReadOnlyList<RelRect> Detect(ImageData image)
		{
			throw new System.NotImplementedException();
		}

		public void Dispose()
		{
			throw new System.NotImplementedException();
		}
	}
}