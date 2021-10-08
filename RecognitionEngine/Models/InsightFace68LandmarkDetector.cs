using RecognitionPrimitives;
using RecognitionPrimitives.Models;
using System;
using System.Collections.Generic;

namespace RecognitionEngine.Models
{
	internal class InsightFace68LandmarkDetector : IFaceLandmarkDetector
	{
		public InsightFace68LandmarkDetector(byte[] modelBytes)
		{

		}

		public IReadOnlyList<IFaceLandmarks> GetFacesLandmarks(object filteredFaces)
		{
			throw new NotImplementedException();
		}

		public void Dispose()
		{
			throw new NotImplementedException();
		}
	}
}