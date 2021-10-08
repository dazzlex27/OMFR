using System;
using System.Threading.Tasks;

namespace Primitives.Logging
{
	public class DummyLogger : ILogger
	{
		public async Task LogDebug(string message)
		{
			await Task.FromResult(0);
		}

		public async Task LogError(string message)
		{
			await Task.FromResult(0);
		}

		public async Task LogException(string message, Exception ex)
		{
			await Task.FromResult(0);
		}

		public async Task LogInfo(string message)
		{
			await Task.FromResult(0);
		}
	}
}