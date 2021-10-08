using Primitives;
using System;

namespace RecognitionPrimitives.Models
{
	public interface IFaceIndexer : IDisposable
	{
		IFaceIndex GetFaceIndex(ImageData faceImage);
	}
}