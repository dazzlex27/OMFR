using Primitives.Logging;
using RecognitionPrimitives;
using System;
using System.Collections.Generic;

namespace RecognitionEngine
{
	public class IndexProcessor
	{
		private readonly ILogger _logger;
		private readonly IFaceIndexComparer _indexComparer;

		public IndexProcessor(ILogger logger, IFaceIndexComparer indexComparer)
		{
			_logger = logger;
			_logger.LogInfo($"Creating index processor for index type {indexComparer.IndexType}...");

			_indexComparer = indexComparer;
		}

		public float MatchOneToOne(IFaceIndex index1, IFaceIndex index2)
		{
			return GetIndexSimilarity(index1, index2);
		}

		public IReadOnlyList<(Guid, float)> MatchOneToManyWithThreshold(IFaceIndex index, Dictionary<Guid, IFaceIndex> listToMatch,
			float threshold)
		{
			var results = new List<(Guid, float)>();
			foreach (var indexToMatch in listToMatch)
			{
				var similarity = GetIndexSimilarity(index, indexToMatch.Value);
				if (similarity < threshold)
					continue;

				results.Add((indexToMatch.Key, similarity));
			}

			return results;
		}

		public IReadOnlyList<float> MatchToMany(IFaceIndex index, IReadOnlyList<IFaceIndex> listToMatch)
		{
			var results = new List<float>();

			foreach (var indexToMatch in  listToMatch)
			{
				var similarity = GetIndexSimilarity(index, indexToMatch);
				results.Add(similarity);
			}

			return results;
		}

		private float GetIndexSimilarity(IFaceIndex index1, IFaceIndex index2)
		{
			if (index1.Version != index2.Version)
				throw new NotImplementedException($"Different versions of indexes ({index1.Version} vs {index2.Version}) are not comparable");

			if (index1.Version != _indexComparer.IndexType)
				throw new NotImplementedException($"Invalid version for comparison: {index1.Version} vs {_indexComparer.IndexType}");

			return _indexComparer.Compare(index1, index2);
		}
	}
}