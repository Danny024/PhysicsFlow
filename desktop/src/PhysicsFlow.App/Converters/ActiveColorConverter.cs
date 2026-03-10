using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace PhysicsFlow.App.Converters;

public class ActiveColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        bool isActive = value is bool b && b;
        return isActive
            ? new SolidColorBrush(Colors.White)
            : new SolidColorBrush(Color.FromRgb(0x88, 0x88, 0x99));
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
