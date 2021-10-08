using System;
using System.Threading.Tasks;

namespace Primitives.Logging
{
	public interface ILogger
	{
		Task LogInfo(string message);

		Task LogError(string message);

		Task LogException(string message, Exception ex);

		Task LogDebug(string message);
	}
}
