namespace RecognitionPrimitives
{
	public interface IFaceIndexComparer
	{
		string IndexType { get; }

		float Compare(IFaceIndex index1, IFaceIndex index2);
	}
}