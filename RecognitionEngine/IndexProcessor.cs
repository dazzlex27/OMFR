using RecognitionPrimitives;

namespace RecognitionEngine
{
	public class IndexProcessor
	{
		public float MatchOneToOne(IFaceIndex index1, IFaceIndex index2)
		{
			return GetIndexSimilarity(index1, index2);
		}

		private float GetIndexSimilarity(IFaceIndex index1, IFaceIndex index2)
		{
			// TODO: implement
			return -1; 
		}
	}
}