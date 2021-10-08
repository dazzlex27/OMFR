using System;
using System.IO;
using System.Threading.Tasks;

namespace Primitives.Logging
{
	public class TxtLogger : ILogger
	{
		private const int MaxLogSizeBytes = 5 * 1000 * 1000;
		private static readonly DateTime StartupTime = DateTime.Now;

		private readonly string _filePath;
		private readonly string _archiveFilePath;

		public TxtLogger(string logName)
		{
			var logFileName = $"{logName}.log";
			var archivedLogFileName = $"{logFileName}1";

			var currentInstanceFolder = StartupTime.ToString("yyyy-MM-dd-HH-mm-ss");
			var logDirectory = Path.Combine(GlobalConstants.AppLogsPath, currentInstanceFolder);
			Directory.CreateDirectory(logDirectory);
			_filePath = Path.Combine(logDirectory, logFileName);
			_archiveFilePath = Path.Combine(logDirectory, archivedLogFileName);
		}

		public async Task LogInfo(string message)
		{
			await WriteMessageToLog("INFO", message);
		}

		public async Task LogError(string message)
		{
			await WriteMessageToLog("ERROR", message);
		}

		public async Task LogException(string message, Exception ex)
		{
			await WriteMessageToLog("EXCEPTION", $"{message} {ex}");

#if DEBUG
			Console.Beep();
#endif
		}

		public async Task LogDebug(string message)
		{
			await WriteMessageToLog("DEBUG", message);
		}

		private async Task WriteMessageToLog(string type, string message)
		{
			var time = DateTime.Now;
			await using (var sw = File.AppendText(_filePath))
			{
				await sw.WriteLineAsync($"{time.ToShortDateString()} {time:HH:mm:ss.fff} {type}: {message}");
				await sw.FlushAsync();
			}

			var logFileInfo = new FileInfo(_filePath);
			var needToArchiveLogFile = logFileInfo.Length > MaxLogSizeBytes;
			if (!needToArchiveLogFile)
				return;

			await ArchiveLogFile();
		}

		private async Task ArchiveLogFile()
		{
			try
			{
				var additionalLogFileInfo = new FileInfo(_archiveFilePath);
				if (additionalLogFileInfo.Exists)
					File.Delete(_archiveFilePath);

				File.Move(_filePath, _archiveFilePath);
			}
			catch (Exception ex)
			{
				await LogException("Failed to archive log file", ex);
			}
		}
	}
}