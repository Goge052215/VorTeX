from manim import *
import logging
import numpy as np

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
            self.logger = logger  # Store the logger

        def construct(self):
            axes = Axes(
                x_range=self.x_range,
                y_range=self.y_range,
                axis_config={"color": BLUE},
                x_length=10,
                y_length=6
            )
            
            try:
                # Log the function string for debugging
                if self.logger:
                    self.logger.debug(f"Evaluating function: {self.func_str}")
                
                # Use a safe eval context
                safe_dict = {"x": 0, "np": np, "e": np.e}
                graph = axes.plot(lambda x: eval(self.func_str, {"x": x, "np": np, "e": np.e}, safe_dict))
                
                # Create animation
                self.play(Create(axes))
                self.play(Create(graph))
            except Exception as e:
                self.add(Text(f"Error: {str(e)}"))
                if self.logger:
                    self.logger.error(f"Error in function visualization: {e}")

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
            
            # Check if the function string is valid
            try:
                # Test evaluation to ensure the function is valid
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