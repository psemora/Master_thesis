# ------------------------------------------------------ #
# -------------------- mplwidget.py -------------------- #
# ------------------------------------------------------ #
from PyQt5.QtWidgets import*

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

#Creates canvas as Matplotlib figure  
class MplWidget(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

#Toolbar to operate with canvas
class NavigationToolbar(NavigationToolbar):
    #Displays only the buttons we need
    NavigationToolbar.toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        # ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
    )