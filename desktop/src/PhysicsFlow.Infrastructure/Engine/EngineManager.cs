using System.Diagnostics;
using System.IO;
using Microsoft.Extensions.Logging;

namespace PhysicsFlow.Infrastructure.Engine;

/// <summary>
/// Manages the lifecycle of the Python PhysicsFlow gRPC engine process.
///
/// On startup: locates the bundled Python environment, starts the gRPC server,
///             waits for the "engine.ready" signal file, then connects.
/// On shutdown: sends SIGTERM and waits for graceful exit (5s timeout).
/// </summary>
public class EngineManager : IAsyncDisposable
{
    private readonly ILogger<EngineManager> _logger;
    private Process? _engineProcess;
    private CancellationTokenSource? _cts;

    // Path to bundled Python (relative to application directory)
    private static readonly string PythonExe = Path.Combine(
        AppDomain.CurrentDomain.BaseDirectory,
        "engine", "python", "python.exe"
    );

    // Fallback: system Python (developer mode)
    private static readonly string FallbackPython = "python";

    private static readonly string EngineModule = "physicsflow.server";
    private static readonly string ReadySignalFile = "engine.ready";

    public int Port { get; private set; } = 50051;
    public bool IsRunning => _engineProcess is { HasExited: false };
    public event EventHandler<EngineStatusEventArgs>? StatusChanged;

    public EngineManager(ILogger<EngineManager> logger)
    {
        _logger = logger;
    }

    // ── Start ─────────────────────────────────────────────────────────────────

    public async Task StartAsync(int port = 50051, CancellationToken ct = default)
    {
        Port = port;
        OnStatusChanged(EngineStatus.Starting);

        // If engine is already running (dev started it manually), skip spawning
        if (await IsEngineAlreadyRunningAsync(ct))
        {
            _logger.LogInformation("Engine already running on port {Port} — skipping auto-start", port);
            OnStatusChanged(EngineStatus.Running);
            return;
        }

        var workDir = GetEngineWorkDir();

        // Clean up any stale ready signal
        var signalPath = Path.Combine(workDir, ReadySignalFile);
        if (File.Exists(signalPath))
            File.Delete(signalPath);

        var python = GetPythonExe(workDir);

        _logger.LogInformation("Starting PhysicsFlow engine on port {Port}", port);
        _logger.LogInformation("Python: {Python}, WorkDir: {WorkDir}", python, workDir);

        _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);

        var psi = new ProcessStartInfo
        {
            FileName = python,
            Arguments = $"-m {EngineModule} --port {port} --log-level INFO",
            WorkingDirectory = workDir,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        _engineProcess = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start Python engine process.");

        // Forward engine stdout/stderr to our logger
        _engineProcess.OutputDataReceived += (_, e) =>
        {
            if (e.Data != null) _logger.LogDebug("[Engine] {Line}", e.Data);
        };
        _engineProcess.ErrorDataReceived += (_, e) =>
        {
            if (e.Data != null) _logger.LogWarning("[Engine ERR] {Line}", e.Data);
        };
        _engineProcess.BeginOutputReadLine();
        _engineProcess.BeginErrorReadLine();

        // Monitor for unexpected exit
        _engineProcess.Exited += OnEngineExited;
        _engineProcess.EnableRaisingEvents = true;

        // Wait for ready signal (up to 60 seconds)
        await WaitForReadyAsync(workDir, TimeSpan.FromSeconds(60), _cts.Token);

        _logger.LogInformation("PhysicsFlow engine ready on port {Port}", port);
        OnStatusChanged(EngineStatus.Running);
    }

    // ── Stop ──────────────────────────────────────────────────────────────────

    public async Task StopAsync()
    {
        if (_engineProcess is null || _engineProcess.HasExited)
            return;

        _logger.LogInformation("Stopping PhysicsFlow engine (PID {Pid})", _engineProcess.Id);

        try
        {
            // Send SIGTERM equivalent (kill gracefully)
            _engineProcess.Kill(entireProcessTree: true);
            await _engineProcess.WaitForExitAsync(new CancellationTokenSource(5_000).Token);
        }
        catch (OperationCanceledException)
        {
            _logger.LogWarning("Engine did not exit within 5s; force-killing.");
            _engineProcess.Kill();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping engine");
        }

        OnStatusChanged(EngineStatus.Stopped);
    }

    // ── Restart ───────────────────────────────────────────────────────────────

    public async Task RestartAsync()
    {
        await StopAsync();
        await Task.Delay(1_000);
        await StartAsync(Port);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static async Task<bool> IsEngineAlreadyRunningAsync(CancellationToken ct)
    {
        // Try REST health endpoint (3 attempts × 500 ms — REST starts slightly after gRPC)
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
        for (int attempt = 0; attempt < 3; attempt++)
        {
            try
            {
                var resp = await http.GetAsync("http://localhost:8000/api/v1/health", ct);
                if (resp.IsSuccessStatusCode) return true;
            }
            catch { }
            if (attempt < 2)
                await Task.Delay(700, ct);
        }

        // Fallback: TCP connect to gRPC port — confirms the gRPC server is up
        try
        {
            using var tcp = new System.Net.Sockets.TcpClient();
            var connect = tcp.ConnectAsync("127.0.0.1", 50051);
            return await Task.WhenAny(connect, Task.Delay(1000, ct)) == connect
                   && tcp.Connected;
        }
        catch
        {
            return false;
        }
    }

    private static string GetPythonExe(string engineWorkDir)
    {
        // 1. Bundled Python (production installer)
        if (File.Exists(PythonExe)) return PythonExe;

        // 2. .venv inside the engine directory (developer workflow)
        var venv = Path.Combine(engineWorkDir, ".venv", "Scripts", "python.exe");
        if (File.Exists(venv)) return venv;

        // 3. System python — last resort
        return FallbackPython;
    }

    private static async Task WaitForReadyAsync(string workDir, TimeSpan timeout, CancellationToken ct)
    {
        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(timeout);

        var signalPath = Path.Combine(workDir, ReadySignalFile);

        while (!timeoutCts.Token.IsCancellationRequested)
        {
            if (File.Exists(signalPath))
            {
                File.Delete(signalPath);
                return;
            }
            await Task.Delay(200, timeoutCts.Token);
        }

        throw new TimeoutException("PhysicsFlow engine did not start within timeout.");
    }

    private static string GetEngineWorkDir()
    {
        // Walk up from the app binary looking for a sibling "engine/" folder
        // that contains pyproject.toml — this is the PhysicsFlow project root.
        var dir = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
        while (dir != null)
        {
            var enginePath = Path.Combine(dir.FullName, "engine");
            if (Directory.Exists(enginePath) &&
                File.Exists(Path.Combine(enginePath, "pyproject.toml")))
                return enginePath;
            dir = dir.Parent;
        }

        // Production: engine/ is bundled next to the executable
        return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "engine");
    }

    private void OnEngineExited(object? sender, EventArgs e)
    {
        if (_engineProcess?.ExitCode != 0)
        {
            _logger.LogError("Engine exited unexpectedly (code {Code})",
                _engineProcess?.ExitCode);
            OnStatusChanged(EngineStatus.Error);
        }
        else
        {
            _logger.LogInformation("Engine stopped cleanly.");
            OnStatusChanged(EngineStatus.Stopped);
        }
    }

    private void OnStatusChanged(EngineStatus status)
    {
        StatusChanged?.Invoke(this, new EngineStatusEventArgs(status));
    }

    public async ValueTask DisposeAsync()
    {
        await StopAsync();
        _engineProcess?.Dispose();
        _cts?.Dispose();
    }
}

public enum EngineStatus { Starting, Running, Stopped, Error }

public class EngineStatusEventArgs(EngineStatus status) : EventArgs
{
    public EngineStatus Status { get; } = status;
}
