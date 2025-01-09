from manim import *
import logging
from typing import Optional, Union, List
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
        def __init__(self, func_str: str, x_range=(-10, 10), **kwargs):
            super().__init__(**kwargs)
            self.func_str = func_str
            self.x_range = x_range

        def construct(self):
            # Create axes
            axes = Axes(
                x_range=self.x_range,
                y_range=(-5, 5),
                axis_config={"color": BLUE},
            )
            
            # Create graph
            try:
                graph = axes.plot(lambda x: eval(self.func_str))
                self.play(Create(axes), Create(graph))
                self.wait()
            except Exception as e:
                self.add(Text(f"Error: {str(e)}"))
                self.wait()

    class DerivativeScene(Scene):
        """Scene for visualizing derivatives."""
        def __init__(self, func_str: str, point: float = 0, **kwargs):
            super().__init__(**kwargs)
            self.func_str = func_str
            self.point = point

        def construct(self):
            # Create axes
            axes = Axes(
                x_range=(-5, 5),
                y_range=(-5, 5),
                axis_config={"color": BLUE},
            )
            
            # Plot original function
            try:
                func = lambda x: eval(self.func_str)
                graph = axes.plot(func)
                
                # Calculate derivative at point
                dx = 0.001
                derivative = (func(self.point + dx) - func(self.point)) / dx
                
                # Create tangent line
                point = axes.c2p(self.point, func(self.point))
                tangent = Line(
                    start=axes.c2p(self.point - 1, func(self.point) - derivative),
                    end=axes.c2p(self.point + 1, func(self.point) + derivative),
                    color=RED
                )
                
                # Animate
                self.play(Create(axes), Create(graph))
                self.play(Create(tangent))
                self.wait()
            except Exception as e:
                self.add(Text(f"Error: {str(e)}"))
                self.wait()

    def visualize_function(self, func_str: str, x_range: tuple = (-10, 10)) -> None:
        """
        Create a visualization of a mathematical function.
        
        Args:
            func_str (str): String representation of the function (e.g., "x**2")
            x_range (tuple): Range for x-axis (default: (-10, 10))
        """
        try:
            self.logger.info(f"Visualizing function: {func_str}")
            scene = self.FunctionScene(func_str, x_range)
            scene.render()
        except Exception as e:
            self.logger.error(f"Error visualizing function: {e}")
            raise

    def visualize_derivative(self, func_str: str, point: float = 0) -> None:
        """
        Create a visualization of a function's derivative at a point.
        
        Args:
            func_str (str): String representation of the function
            point (float): Point at which to show the derivative
        """
        try:
            self.logger.info(f"Visualizing derivative of {func_str} at x={point}")
            scene = self.DerivativeScene(func_str, point)
            scene.render()
        except Exception as e:
            self.logger.error(f"Error visualizing derivative: {e}")
            raise