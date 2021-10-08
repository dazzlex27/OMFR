using Primitives;
using Primitives.Structs;
using RecognitionPrimitives.Models;
using System;
using System.Collections.Generic;

namespace RecognitionEngine.Models
{
	internal class ConvNetFaceFilter : IFaceFilter
	{
		public ConvNetFaceFilter(byte[] modelBytes)
		{

		}

		public void Dispose()
		{
			throw new NotImplementedException();
		}

		public IReadOnlyList<RelRect> GetFilteredFaces(ImageData image, IReadOnlyList<RelRect> detectedFaces)
		{
			throw new NotImplementedException();
		}
	}
}