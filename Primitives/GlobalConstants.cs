using System;
using System.Diagnostics;
using System.IO;

namespace Primitives
{
	public static class GlobalConstants
	{
		public const string DataDirectoryName = "data";

		public static readonly string AppHeaderString;
		public static readonly string AppLogsPath;
		public static readonly string ConfigFileName;

		static GlobalConstants()
		{
			const string appVersion = "v0.1";
			var appTitle = Process.GetCurrentProcess().ProcessName;
			AppHeaderString = $@"{appTitle} {appVersion}";

			var commonFolderPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
			var appDataPath = Path.Combine(commonFolderPath, appTitle);
			AppLogsPath = Path.Combine(appDataPath, "logs");

			var appConfigPath = Path.Combine(appDataPath, "config");
			ConfigFileName = Path.Combine(appConfigPath, "main.cfg");
		}
	}
}