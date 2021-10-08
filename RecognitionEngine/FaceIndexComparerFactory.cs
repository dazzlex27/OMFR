using RecognitionPrimitives;
using System;

namespace RecognitionEngine
{
	public static class FaceIndexComparerFactory
	{
		public static IFaceIndexComparer CreateFromIndexType(string type)
		{
			switch (type)
			{
				case "arc50_1":
					return new ArcF50FaceIndexComparer();
				default:
					throw new NotImplementedException($"Index type {type} is not supported");
			}
		}
	}
}