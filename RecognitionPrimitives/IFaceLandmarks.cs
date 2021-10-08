using Primitives.Structs;
using System.Collections.Generic;

namespace RecognitionPrimitives
{
	public interface IFaceLandmarks
	{
		IReadOnlyList<RelPoint> Points { get; }

		RelPoint LeftEye { get; }

		RelPoint RightEye { get; }

		RelPoint CenterNose { get; }

		RelPoint LeftMouth { get; }

		RelPoint RightMouth { get; }
	}
}