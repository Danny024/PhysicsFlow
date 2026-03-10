using MahApps.Metro.Controls;
using PhysicsFlow.ViewModels;

namespace PhysicsFlow.App;

public partial class MainWindow : MetroWindow
{
    public MainWindow(MainWindowViewModel viewModel)
    {
        InitializeComponent();
        DataContext = viewModel;
    }
}
