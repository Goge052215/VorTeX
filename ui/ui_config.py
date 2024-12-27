class UiConfig:
    def config_button(self, theme_callback, legend_callback):
        """Configure button settings with callbacks.
        
        Args:
            theme_callback: Function to handle theme menu display
            legend_callback: Function to handle legend display
        """
        self.button_configs = {
            'theme': {
                'text': 'Themes',
                'size': (80, 30),
                'callback': theme_callback
            },
            'legend': {
                'text': 'Show Legend',
                'size': (120, 30),
                'callback': legend_callback
            }
        }
        
        self.button_style = """
            QPushButton {
                background-color: #3d59a1;
                color: #ffffff;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2ac3de;
            }
        """
    def mode_config(self, calculator):
        """Configure UI modes with calculator widget references.
        
        Args:
            calculator: Instance of CalculatorApp containing the widgets
        """
        self.mode_configs = {
            'Matrix': {
                'show': [
                    calculator.matrix_input,
                    calculator.label_matrix_op,
                    calculator.combo_matrix_op,
                    calculator.store_matrix_button,
                    calculator.recall_matrix_button
                ],
                'hide': [
                    calculator.label_formula,
                    calculator.entry_formula,
                    calculator.calculate_button
                ],
                'dimensions': (450, 600)  # (height, width)
            },
            'default': {
                'show': [
                    calculator.label_formula,
                    calculator.entry_formula,
                    calculator.calculate_button
                ],
                'hide': [
                    calculator.matrix_input,
                    calculator.label_matrix_op,
                    calculator.combo_matrix_op,
                    calculator.store_matrix_button,
                    calculator.recall_matrix_button
                ],
                'dimensions': (500, 700)  # (height, width)
            }
        }

