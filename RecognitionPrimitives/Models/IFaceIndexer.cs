using Primitives;
using System;

namespace RecognitionPrimitives.Models
{
	public interface IFaceIndexer : IDisposable
	{
		string IndexType { get; }

		IFaceIndex GetFaceIndex(ImageData faceImage);
	}
}