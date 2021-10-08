using System;
using System.Threading.Tasks;

namespace Primitives.Logging
{
	public class ConsoleLogger : ILogger
	{
		public async Task LogDebug(string message)
		{
			Console.ForegroundColor = ConsoleColor.Blue;
			Console.WriteLine($"DEBUG: {message}");
			Console.ResetColor();
			await Task.FromResult(0);
		}

		public async Task LogError(string message)
		{
			Console.ForegroundColor = ConsoleColor.DarkMagenta;
			Console.WriteLine($"ERROR: {message}");
			Console.ResetColor();
			await Task.FromResult(0);
		}

		public async Task LogException(string message, Exception ex)
		{
			Console.ForegroundColor = ConsoleColor.Red;
			Console.WriteLine($"EXCEPTION: {message}{Environment.NewLine}{ex}");
			Console.ResetColor();
			await Task.FromResult(0);
		}

		public async Task LogInfo(string message)
		{
			Console.ForegroundColor = ConsoleColor.Yellow;
			Console.WriteLine($"INFO: {message}");
			Console.ResetColor();
			await Task.FromResult(0);
		}
	}
}