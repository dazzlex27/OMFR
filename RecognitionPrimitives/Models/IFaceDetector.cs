using Primitives;
using Primitives.Structs;
using System;
using System.Collections.Generic;

namespace RecognitionPrimitives.Models
{
	public interface IFaceDetector : IDisposable
	{
		IReadOnlyList<RelRect> Detect(ImageData image);
	}
}