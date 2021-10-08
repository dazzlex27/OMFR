using Primitives;
using RecognitionPrimitives;
using RecognitionPrimitives.Models;
using System;

namespace RecognitionEngine.Models
{
	internal class ArcFace50FaceIndexer : IFaceIndexer
	{
		public ArcFace50FaceIndexer(byte[] genderAgeClassifierBytes)
		{

		}

		public string IndexType => "arc50_1";

		public IFaceIndex GetFaceIndex(ImageData faceImage)
		{
			throw new NotImplementedException();
		}

		public void Dispose()
		{
			throw new NotImplementedException();
		}
	}
}