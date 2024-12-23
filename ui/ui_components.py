from PyQt5.QtWidgets import QPushButton, QLabel, QComboBox, QHBoxLayout, QVBoxLayout, QMenu, QTextEdit
from PyQt5.QtCore import Qt
from themes.theme_manager import get_tokyo_night_theme, get_aura_theme, get_light_theme
from ui.legend_window import LegendWindow
from ui.ui_config import UiConfig
from PyQt5.QtGui import QFont

class UIComponents:
    def __init__(self, parent):
        """
        Initialize UIComponents with a reference to the parent CalculatorApp.
        
        Args:
            parent (CalculatorApp): The instance of the main calculator application.
        """
        self.parent = parent

    def init_ui(self):
        """Initialize the calculator's user interface."""
        self._setup_window()
        main_layout = QVBoxLayout()
        
        # Initialize all UI components using helper methods
        layouts = {
            'top': self._create_top_buttons(),
            'mode': self._create_mode_selection(),
            'angle': self._create_angle_selection(),
            'matrix': self._create_matrix_components(),
            'formula': self._create_formula_components(),
            'result': self._create_result_components()
        }
        
        # Add all layouts to main layout in order
        for layout in layouts.values():
            if isinstance(layout, (list, tuple)):
                for item in layout:
                    if isinstance(item, QHBoxLayout):
                        main_layout.addLayout(item)
                    else:
                        main_layout.addWidget(item)
            else:
                if isinstance(layout, QHBoxLayout):
                    main_layout.addLayout(layout)
                else:
                    main_layout.addWidget(layout)
        
        self.parent.setLayout(main_layout)

    def _setup_window(self):
        """Set up the main window properties."""
        self.parent.setWindowTitle('Scientific Calculator')
        self.parent.setGeometry(100, 100, 600, 450)

    def _create_top_buttons(self):
        """Create theme and legend buttons."""
        top_layout = QHBoxLayout()
        
        # Create UiConfig instance and get configurations
        ui_config = UiConfig()
        ui_config.config_button(
            theme_callback=self.parent.show_theme_menu,
            legend_callback=self.parent.show_legend
        )
        
        top_layout.addStretch()
        for btn_name, config in ui_config.button_configs.items():
            button = QPushButton(config['text'])
            button.setFixedSize(*config['size'])
            button.setStyleSheet(ui_config.button_style)
            # Connect to the parent's callback methods
            button.clicked.connect(config['callback'])
            setattr(self.parent, f"{btn_name}_button", button)
            top_layout.addWidget(button)
        
        return top_layout

    def _create_mode_selection(self):
        """Create input mode selection components."""
        mode_layout = QHBoxLayout()
        self.parent.label_mode = QLabel('Input Mode:')
        self.parent.label_mode.setFixedWidth(100)
        
        self.parent.combo_mode = QComboBox()
        self.parent.combo_mode.addItems(['LaTeX', 'MATLAB', 'Matrix'])
        self.parent.combo_mode.setFixedWidth(100)
        self.parent.combo_mode.currentTextChanged.connect(self.parent.on_mode_changed)
        
        mode_layout.addWidget(self.parent.label_mode)
        mode_layout.addWidget(self.parent.combo_mode)
        mode_layout.addStretch()
        return mode_layout

    def _create_angle_selection(self):
        """Create angle mode selection components."""
        angle_layout = QHBoxLayout()
        self.parent.label_angle = QLabel('Angle Mode:')
        self.parent.label_angle.setFixedWidth(100)
        
        self.parent.combo_angle = QComboBox()
        self.parent.combo_angle.addItems(['Degree', 'Radian'])
        self.parent.combo_angle.setFixedWidth(100)
        
        angle_layout.addWidget(self.parent.label_angle)
        angle_layout.addWidget(self.parent.combo_angle)
        angle_layout.addStretch()
        return angle_layout

    def _create_matrix_components(self):
        """Create matrix operation components."""
        matrix_layout = QHBoxLayout()
        
        # Create Store Matrix Button
        self.parent.store_matrix_button = QPushButton('Store Matrix')
        self.parent.store_matrix_button.setFixedSize(120, 30)
        self.parent.store_matrix_button.clicked.connect(self.parent.store_matrix)  # Connect to parent
        
        # Create Recall Matrix Button
        self.parent.recall_matrix_button = QPushButton('Recall Matrix')
        self.parent.recall_matrix_button.setFixedSize(120, 30)
        self.parent.recall_matrix_button.clicked.connect(self.parent.recall_matrix)  # Connect to parent
        
        matrix_layout.addWidget(self.parent.store_matrix_button)
        matrix_layout.addWidget(self.parent.recall_matrix_button)
        matrix_layout.addStretch()
        
        return matrix_layout

    def _create_formula_components(self):
        """Create formula input components."""
        formula_layout = QHBoxLayout()
        self.parent.entry_formula = QTextEdit()
        self.parent.entry_formula.setPlaceholderText(self.parent.PLACEHOLDER_TEXT)
        self.parent.entry_formula.setFont(self.parent.FORMULA_FONT)
        formula_layout.addWidget(self.parent.entry_formula)
        return formula_layout

    def _create_result_components(self):
        """Create result display components."""
        result_layout = QHBoxLayout()
        self.parent.result_label = QLabel('Result:')
        self.parent.result_label.setFont(QFont("Arial", 13, QFont.Bold))
        result_layout.addWidget(self.parent.result_label)
        result_layout.addStretch()
        return result_layout

    # You can add additional UI component creation methods here if needed.