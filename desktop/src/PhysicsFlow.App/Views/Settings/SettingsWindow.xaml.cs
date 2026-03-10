using System.Windows;
using PhysicsFlow.ViewModels;

namespace PhysicsFlow.App.Views.Settings;

public partial class SettingsWindow : Window
{
    public SettingsWindow(SettingsViewModel viewModel)
    {
        InitializeComponent();
        DataContext = viewModel;
        viewModel.SettingsSaved += (_, _) => Close();
    }

    private void OnCancel(object sender, RoutedEventArgs e) => Close();
}
