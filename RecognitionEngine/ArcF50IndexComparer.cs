using RecognitionPrimitives;
using System;

namespace RecognitionEngine
{
	internal class ArcF50FaceIndexComparer : IFaceIndexComparer
	{
		public string IndexType => "arc50_1";

		public float Compare(IFaceIndex index1, IFaceIndex index2)
		{
			throw new NotImplementedException();
		}
	}
}