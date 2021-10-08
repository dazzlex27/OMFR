using System.IO;

namespace RecognitionEngine
{
	public class ModelFromDiskLoader : IModelLoader
	{
		public byte[] Load(string path)
		{
			return File.ReadAllBytes(path);
		}
	}
}