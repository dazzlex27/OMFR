using Primitives;
using Primitives.Structs;
using System;
using System.Collections.Generic;

namespace RecognitionPrimitives.Models
{
	public interface IFaceNormalizer : IDisposable
	{
		IReadOnlyList<ImageData> Normalize(IReadOnlyList<RelRect> filteredFaces, IReadOnlyList<IFaceLandmarks> facesLandmarks);
	}
}