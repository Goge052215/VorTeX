from manim import *
import logging
import numpy as np
import re

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
        def __init__(self, func_str: str, x_range=(-10, 10), y_range=(-5, 5), logger=None, **kwargs):
            super().__init__(**kwargs)
            self.func_str = func_str
            self.x_range = x_range
            self.y_range = y_range
            self.logger = logger
            
            if self.logger:
                self.logger.debug(f"Creating FunctionScene with function: {func_str}")

        def construct(self):
            # Create the axes
            axes = Axes(
                x_range=self.x_range,
                y_range=self.y_range,
                axis_config={"color": BLUE},
                x_length=10,
                y_length=6
            )
            
            try:
                if self.logger:
                    self.logger.debug(f"Plotting function: {self.func_str}")
                
                # Create a safe evaluation function
                def safe_eval(x):
                    safe_dict = {
                        "x": x,
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
                    return eval(self.func_str, {"__builtins__": {}}, safe_dict)
                
                # Create the graph
                graph = axes.plot(safe_eval, color=WHITE)
                
                # Add labels
                x_label = axes.get_x_axis_label("x")
                y_label = axes.get_y_axis_label("y")
                
                # Create animation
                self.play(Create(axes), Create(x_label), Create(y_label))
                self.play(Create(graph))
                
                # Pause at the end to show the complete graph
                self.wait(1)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in function visualization: {e}")
                self.add(Text(f"Error: {str(e)}").scale(0.5))

    def visualize_function(self, func_str: str, x_range: tuple = (-10, 10), y_range: tuple = (-5, 5)):
        """
        Create a visualization of a mathematical function.
        
        Args:
            func_str (str): String representation of the function (e.g., "x**2")
            x_range (tuple): Range for x-axis (default: (-10, 10))
            y_range (tuple): Range for y-axis (default: (-5, 5))
        """
        try:
            self.logger.info(f"Visualizing function: {func_str}")
            
            # Clean up the function string
            func_str = func_str.strip()
            if not func_str:
                raise ValueError("Empty function string")
            
            # Handle implicit multiplication (e.g., "1/2 x" -> "1/2*x")
            func_str = re.sub(r'(\d+)\s+([a-zA-Z])', r'\1*\2', func_str)
            func_str = re.sub(r'(\d+)/(\d+)\s+([a-zA-Z])', r'(\1/\2)*\3', func_str)
            
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
            
            self.logger.debug(f"Processed function string: {func_str}")
            
            # Define a safe evaluation context with common mathematical functions
            safe_dict = {
                "x": 0,
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
            
            # Test evaluation with a sample value
            x = 0  # Test value
            try:
                eval(func_str, {"__builtins__": {}}, {**safe_dict, "x": x})
            except Exception as e:
                self.logger.error(f"Invalid function syntax: {e}")
                raise ValueError(f"Invalid function syntax: {e}")
            
            # Create and render the scene
            scene = self.FunctionScene(func_str, x_range, y_range, logger=self.logger)
            scene.render()
            
        except Exception as e:
            self.logger.error(f"Error visualizing function: {e}")
            raise