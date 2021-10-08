using Primitives;
using Primitives.Structs;
using RecognitionPrimitives;
using RecognitionPrimitives.Models;
using System.Collections.Generic;

namespace RecognitionEngine.Models
{
	internal class InsightFaceNormalizer : IFaceNormalizer
	{
		public InsightFaceNormalizer()
		{
		}

		public IReadOnlyList<ImageData> Normalize(IReadOnlyList<RelRect> filteredFaces, IReadOnlyList<IFaceLandmarks> facesLandmarks)
		{
			throw new System.NotImplementedException();
		}

		public void Dispose()
		{
			throw new System.NotImplementedException();
		}
	}
}