using Primitives;

namespace RecognitionPrimitives
{
	public interface IFaceInfo
	{
		ImageData FaceImage { get; }

		IFaceIndex FaceIndex { get; }

		FaceMaskStatus MaskStatus { get; }

		FaceGender Gender { get; }

		int Age { get; }
	}
}