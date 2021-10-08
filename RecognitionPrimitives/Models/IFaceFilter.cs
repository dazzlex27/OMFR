using Primitives;
using Primitives.Structs;
using System;
using System.Collections.Generic;

namespace RecognitionPrimitives.Models
{
	public interface IFaceFilter : IDisposable
	{
		IReadOnlyList<RelRect> GetFilteredFaces(ImageData image, IReadOnlyList<RelRect> detectedFaces);
	}
}