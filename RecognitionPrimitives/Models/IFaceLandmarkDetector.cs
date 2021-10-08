using Primitives;
using System;
using System.Collections.Generic;

namespace RecognitionPrimitives.Models
{
	public interface IFaceLandmarkDetector : IDisposable
	{
		IReadOnlyList<IFaceLandmarks> GetFacesLandmarks(object filteredFaces);
	}
}