using Primitives;

namespace RecognitionPrimitives
{
	internal class FaceInfo : IFaceInfo
	{
		public FaceInfo(ImageData faceImage, IFaceIndex faceIndex, FaceMaskStatus maskStatus, FaceGender gender, int age)
		{
			FaceImage = faceImage;
			FaceIndex = faceIndex;
			MaskStatus = maskStatus;
			Gender = gender;
			Age = age;
		}

		public ImageData FaceImage { get; }

		public IFaceIndex FaceIndex { get; }

		public FaceMaskStatus MaskStatus { get; }

		public FaceGender Gender { get; }

		public int Age { get; }
	}
}