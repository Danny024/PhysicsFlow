using System.Windows;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Serilog;
using PhysicsFlow.Infrastructure.Engine;
using PhysicsFlow.Infrastructure.Data;
using PhysicsFlow.ViewModels;

namespace PhysicsFlow.App;

public partial class App : Application
{
    private IHost? _host;

    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        // Configure Serilog structured logging
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .WriteTo.File(
                path: System.IO.Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "PhysicsFlow", "logs", "physicsflow_.log"),
                rollingInterval: RollingInterval.Day,
                retainedFileCountLimit: 7)
            .WriteTo.Console()
            .CreateLogger();

        Log.Information("PhysicsFlow {Version} starting", GetType().Assembly.GetName().Version);

        // Build DI host
        _host = Host.CreateDefaultBuilder()
            .UseSerilog()
            .ConfigureServices(ConfigureServices)
            .Build();

        await _host.StartAsync();

        // Show splash / main window
        var mainWindow = _host.Services.GetRequiredService<MainWindow>();
        mainWindow.Show();
    }

    private static void ConfigureServices(IServiceCollection services)
    {
        // Infrastructure
        services.AddSingleton<EngineManager>();
        services.AddSingleton<GrpcEngineClient>();
        services.AddSingleton<OllamaAgentClient>();
        services.AddSingleton<AppDbService>();

        // ViewModels
        services.AddSingleton<MainWindowViewModel>();
        services.AddTransient<DashboardViewModel>();
        services.AddTransient<ProjectSetupViewModel>();
        services.AddTransient<TrainingViewModel>();
        services.AddTransient<HistoryMatchingViewModel>();
        services.AddTransient<ForecastViewModel>();
        services.AddSingleton<AIAssistantViewModel>();

        // Views
        services.AddSingleton<MainWindow>();
    }

    protected override async void OnExit(ExitEventArgs e)
    {
        if (_host != null)
        {
            var engineManager = _host.Services.GetRequiredService<EngineManager>();
            await engineManager.StopAsync();
            await _host.StopAsync();
            _host.Dispose();
        }

        Log.CloseAndFlush();
        base.OnExit(e);
    }
}
