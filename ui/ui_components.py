from PyQt5.QtWidgets import (
    QPushButton, QLabel, QComboBox, QHBoxLayout, QVBoxLayout, QMenu, QTextEdit,
    QSizePolicy, QInputDialog, QMessageBox, QScrollArea, QSpacerItem
)
from PyQt5.QtGui import QFont, QTextOption, QIcon
from PyQt5.QtCore import Qt
from themes.theme_manager import get_tokyo_night_theme, get_aura_theme, get_light_theme
from ui.legend_window import LegendWindow
from ui.ui_config import UiConfig
from PyQt5.QtGui import QFont
from ui.visualization_window import VisualizationWindow

class UIComponents:
    def __init__(self, parent):
        """
        Initialize UIComponents with a reference to the parent CalculatorApp.
        
        Args:
            parent (CalculatorApp): The instance of the main calculator application.
        """
        self.parent = parent
        self.visualization_window = None

        self.parent.PLACEHOLDER_TEXT = 'Enter Simplified LaTeX expression, e.g., 5C2 + sin(pi/2)\n' \
            'Or MATLAB expression, e.g., nchoosek(5,2) + sin(pi/2)'
        self.parent.FORMULA_FONT = QFont("Monaspace Neon", 14)

    def init_ui(self):
        """Initialize the calculator's user interface."""
        self._setup_window()
        main_layout = QVBoxLayout()
        
        layouts = {
            'top': self._create_top_buttons(),
            'mode': self._create_mode_selection(),
            'angle': self._create_angle_selection(),
            'matrix': self._create_matrix_components(),
            'formula': self._create_formula_components(),
            'result': self._create_result_components()
        }
        
        vertical_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Fixed)
        
        for key, layout in layouts.items():
            if isinstance(layout, (QHBoxLayout, QVBoxLayout)):
                main_layout.addLayout(layout)
            else:
                main_layout.addWidget(layout)
            if key == 'angle':
                main_layout.addSpacerItem(vertical_spacer)
        
        self.parent.setLayout(main_layout)

    def _setup_window(self):
        self.parent.setWindowTitle('VorTeX Calculator')
        self.parent.setGeometry(100, 100, 700, 550)

    def _create_top_buttons(self):
        top_layout = QHBoxLayout()
        
        ui_config = UiConfig()
        ui_config.config_button(
            settings_callback=self.parent.show_settings,
            legend_callback=self.parent.show_legend
        )
        
        top_layout.addStretch()

        for btn_name, config in ui_config.button_configs.items():
            button = QPushButton(config['text'])
            button.setFixedSize(*config['size'])
            button.setStyleSheet(ui_config.button_style)

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
        self.parent.combo_mode.addItems(['LaTeX', 'MATLAB', 'SymPy', 'Matrix'])
        self.parent.combo_mode.setFont(QFont("Monaspace Neon", 12))
        self.parent.combo_mode.setFixedWidth(150)
        self.parent.combo_mode.setFixedHeight(30)
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
        self.parent.combo_angle.setFont(QFont("Monaspace Neon", 12))
        self.parent.combo_angle.setFixedWidth(150)
        self.parent.combo_angle.setFixedHeight(30)
        
        angle_layout.addWidget(self.parent.label_angle)
        angle_layout.addWidget(self.parent.combo_angle)
        angle_layout.addStretch()
        angle_layout.addSpacing(10)
        return angle_layout
    

    def _create_matrix_components(self):
        matrix_layout = QVBoxLayout()
        
        self.parent.matrix_input = QTextEdit()
        self.parent.matrix_input.setPlaceholderText(
            "Enter matrix in MATLAB format, e.g., [1 2; 3 4]\n"
            "Or [1, 2; 3, 4] for comma-separated values"
        )
        self.parent.matrix_input.setFont(QFont("Monaspace Neon", 14))
        self.parent.matrix_input.setFixedHeight(100)
        self.parent.matrix_input.hide()
        
        operation_layout = QHBoxLayout()
        self.parent.label_matrix_op = QLabel('Matrix Operation:')
        self.parent.label_matrix_op.setFixedWidth(120)
        self.parent.combo_matrix_op = QComboBox()
        self.parent.combo_matrix_op.addItems([
            'Determinant', 'Inverse', 'Eigenvalues', 'Rank',
            'Multiply', 'Add', 'Subtract', 'Divide', 'Differentiate'
        ])
        self.parent.combo_matrix_op.setFixedWidth(130)
        self.parent.combo_matrix_op.hide()
        self.parent.label_matrix_op.hide()
        
        operation_layout.addWidget(self.parent.label_matrix_op)
        operation_layout.addWidget(self.parent.combo_matrix_op)
        operation_layout.addStretch()
        
        button_layout = QHBoxLayout()
        
        self.parent.store_matrix_button = QPushButton('Store Matrix')
        self.parent.store_matrix_button.setFixedSize(120, 30)
        self.parent.store_matrix_button.clicked.connect(self.parent.store_matrix)
        self.parent.store_matrix_button.hide()
        
        self.parent.recall_matrix_button = QPushButton('Recall Matrix')
        self.parent.recall_matrix_button.setFixedSize(120, 30)
        self.parent.recall_matrix_button.clicked.connect(self.parent.recall_matrix)
        self.parent.recall_matrix_button.hide()
        
        self.parent.calculate_matrix_button = QPushButton('Calculate Matrix')
        self.parent.calculate_matrix_button.setFixedSize(180, 30)
        self.parent.calculate_matrix_button.clicked.connect(self.parent.calculate_matrix)
        self.parent.calculate_matrix_button.hide()
        
        button_layout.addWidget(self.parent.store_matrix_button)
        button_layout.addWidget(self.parent.recall_matrix_button)
        button_layout.addWidget(self.parent.calculate_matrix_button)
        button_layout.addStretch()
        
        matrix_layout.addWidget(self.parent.matrix_input)
        matrix_layout.addLayout(operation_layout)
        matrix_layout.addLayout(button_layout)
        
        # Visibility managed by mode_config
        self.parent.store_matrix_button.hide()
        self.parent.recall_matrix_button.hide()
        self.parent.calculate_matrix_button.hide()
        return matrix_layout

    def _create_formula_components(self):
        """Create formula input components with vertical layout."""
        formula_layout = QHBoxLayout()
        formula_layout.setContentsMargins(0, 0, 0, 0)
        formula_layout.setSpacing(0)
        
        self.parent.label_formula = QLabel('Math Input:')
        self.parent.label_formula.setFont(QFont("Arial", 13, QFont.Bold))
        self.parent.label_formula.setFixedWidth(100)
        self.parent.label_formula.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        input_button_layout = QVBoxLayout()
        input_button_layout.setContentsMargins(0, 0, 0, 0)
        input_button_layout.setSpacing(20)
        
        self.parent.entry_formula = QTextEdit()
        self.parent.entry_formula.setPlaceholderText(self.parent.PLACEHOLDER_TEXT)
        self.parent.entry_formula.setFont(self.parent.FORMULA_FONT)
        self.parent.entry_formula.setFixedHeight(100)
        self.parent.entry_formula.setStyleSheet("QTextEdit { margin-top: 0px; padding-top: 0px; }")
        
        self.parent.calculate_button = QPushButton('Calculate')
        self.parent.calculate_button.clicked.connect(self.parent.calculate)
        self.parent.calculate_button.setFixedHeight(30)
        
        self.parent.visualize_button = QPushButton('Visualize')
        self.parent.visualize_button.clicked.connect(self.handle_visualization)  # Connect to the new method
        self.parent.visualize_button.setFixedHeight(30)
        
        input_button_layout.addWidget(self.parent.entry_formula)
        input_button_layout.addWidget(self.parent.calculate_button)
        input_button_layout.addWidget(self.parent.visualize_button)
        
        formula_layout.addWidget(self.parent.label_formula)
        formula_layout.addLayout(input_button_layout)
        
        return formula_layout

    def _create_result_components(self):
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(0, 10, 0, 0)
        
        header_layout = QHBoxLayout()
        self.parent.result_label = QLabel('Result:')
        self.parent.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        header_layout.addWidget(self.parent.result_label)
        header_layout.addStretch()
        
        self.parent.result_display = QLabel()
        self.parent.result_display.setFont(QFont("Monaspace Neon", 14))
        self.parent.result_display.setWordWrap(True)
        self.parent.result_display.setTextInteractionFlags(Qt.TextSelectableByMouse)  # Make text selectable
        self.parent.result_display.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 5px;
            }
        """)
        
        result_layout.addLayout(header_layout)
        result_layout.addWidget(self.parent.result_display)
        
        return result_layout

    def handle_visualization(self):
        if self.visualization_window is None:
            self.visualization_window = VisualizationWindow(self.parent)
        
        self.visualization_window.show()
        self.parent.handle_visualization()
