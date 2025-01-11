from manim import *
import logging
import numpy as np
import re
from sympy import Symbol, sympify

class MathVisualizer:
    """Handles mathematical visualizations using Manim."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._configure_logger()

    def _configure_logger(self):
        """Configure logging for the visualizer."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    class FunctionScene(Scene):
        """Base scene for function visualization."""
        def __init__(self, func_str, x_range=(-10, 10), y_range=(-5, 5), logger=None):
            super().__init__()
            self.func_str = func_str
            self.x_range = x_range
            self.y_range = y_range
            self.logger = logger or logging.getLogger(__name__)

        def construct(self):
            try:
                # Create the axes with larger dimensions
                axes = Axes(
                    x_range=[self.x_range[0], self.x_range[1], 1],
                    y_range=[self.y_range[0], self.y_range[1], 1],
                    axis_config={
                        "color": BLUE,
                        "stroke_width": 2,
                        "include_numbers": True,
                        "numbers_to_exclude": [],
                        "font_size": 24  # Increased font size for better visibility
                    },
                    x_length=12,
                    y_length=8,
                    tips=True
                ).scale(1.2)  # Scale the entire axes after creation

                # Center the axes in the frame
                axes.center()
                
                # Add labels with larger size
                x_label = axes.get_x_axis_label("x").scale(1.2)
                y_label = axes.get_y_axis_label("y").scale(1.2)

                # Convert the function string to a lambda function
                # Replace ^ with ** for Python syntax
                func_str = self.func_str.replace('^', '**')
                try:
                    # Create a safe lambda function
                    x = Symbol('x')
                    expr = sympify(self.func_str)
                    func = lambda x: float(expr.subs('x', x))
                except Exception as e:
                    self.logger.error(f"Error parsing function: {e}")
                    return

                # Create the graph with discontinuity handling
                try:
                    graph = axes.plot(
                        lambda x: func(x),
                        color=YELLOW,
                        x_range=[self.x_range[0], self.x_range[1], 0.01],
                        use_smoothing=True,
                        discontinuities=[],
                        dt=0.01
                    )

                    # Create labels for the graph
                    graph_label = MathTex(self.func_str).scale(1.2).next_to(graph, UP)

                    # Add all elements to the scene with animations
                    self.play(Create(axes), run_time=1)
                    self.play(Create(graph), run_time=2)
                    self.play(Write(graph_label), run_time=1)
                    
                    # Add a pause at the end
                    self.wait(2)

                except Exception as e:
                    self.logger.error(f"Error creating graph: {e}")
                    error_text = Text(f"Error: {str(e)}", color=RED).scale(0.8)
                    self.add(error_text)
                    self.wait(2)

            except Exception as e:
                self.logger.error(f"Error in visualization: {e}")
                error_text = Text("Error visualizing function", color=RED)
                self.add(error_text)
                self.wait(2)

    def visualize_function(self, func_str: str, x_range: tuple = (-10, 10), y_range: tuple = (-5, 5)):
        """
        Create a visualization of a mathematical function.
        """
        try:
            self.logger.info(f"Visualizing function: {func_str}")
            
            # Clean up the function string
            func_str = func_str.strip()
            if not func_str:
                raise ValueError("Empty function string")
            
            # Handle implicit multiplication and clean up the expression
            func_str = re.sub(r'(\d+)\s*([a-zA-Z])', r'\1*\2', func_str)  # Convert "2x" to "2*x"
            func_str = re.sub(r'(\d+)/(\d+)\s*([a-zA-Z])', r'(\1/\2)*\3', func_str)
            func_str = re.sub(r'\s+', '', func_str)  # Remove all whitespace
            
            # Replace common mathematical notations
            replacements = {
                '^': '**',
                'sin': 'np.sin',
                'cos': 'np.cos',
                'tan': 'np.tan',
                'exp': 'np.exp',
                'log': 'np.log',
                'ln': 'np.log',
                'sqrt': 'np.sqrt',
                'pi': 'np.pi'
            }
            
            for old, new in replacements.items():
                func_str = func_str.replace(old, new)
            
            # Ensure the expression contains 'x'
            if 'x' not in func_str:
                func_str = f"{func_str}+0*x"  # Add a zero term with x for constant functions
                
            self.logger.debug(f"Processed function string: {func_str}")
            
            # Define a safe evaluation context
            safe_dict = {
                "x": np.linspace(-1, 1, 10),  # Test array
                "np": np,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "exp": np.exp,
                "log": np.log,
                "sqrt": np.sqrt,
                "pi": np.pi,
                "e": np.e
            }
            
            # Test evaluation with numpy array
            try:
                eval(func_str, {"__builtins__": {}}, safe_dict)
            except Exception as e:
                self.logger.error(f"Invalid function syntax: {e}")
                raise ValueError(f"Invalid function syntax: {e}")
            
            # Create and render the scene
            scene = self.FunctionScene(func_str, x_range, y_range, logger=self.logger)
            scene.render()
            
        except Exception as e:
            self.logger.error(f"Error visualizing function: {e}")
            raise