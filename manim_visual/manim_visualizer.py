from manim import *
import logging
import numpy as np
import re
from sympy import Symbol, sympify
import os

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

    def install_latex_dependencies(self):
        """Manually install LaTeX dependencies when needed."""
        try:
            import subprocess
            import os
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            install_script = os.path.join(script_dir, 'tex_pack_install.bash')
            
            if not os.path.exists(install_script):
                self.logger.error("LaTeX installation script not found")
                return
            
            # Make the script executable
            os.chmod(install_script, 0o755)
            
            # Run the installation script
            result = subprocess.run([install_script], 
                                 capture_output=True, 
                                 text=True)
            
            if result.returncode != 0:
                self.logger.error(f"LaTeX installation failed: {result.stderr}")
            else:
                self.logger.info("LaTeX dependencies checked/installed successfully")
                
        except Exception as e:
            self.logger.error(f"Error checking LaTeX dependencies: {e}")

    class FunctionScene(Scene):
        """Scene for visualizing mathematical functions with manim."""
        def __init__(self, func_str, x_range=(-10, 10), y_range=(-5, 5), display_text=None, logger=None):
            """Initialize the function scene with the provided parameters.
            
            Args:
                func_str (str): The function string to visualize.
                x_range (tuple): The range for x-axis.
                y_range (tuple): The range for y-axis.
                display_text (str): Optional text to display instead of the function.
                logger: Logger for debug messages.
            """
            super().__init__()
            self.func_str = func_str
            self.x_range = x_range
            self.y_range = y_range
            self.display_text = display_text if display_text else func_str
            self.logger = logger

        def construct(self):
            """Construct the scene with function graph and labels."""
            # Create axes
            axes = Axes(
                x_range=[self.x_range[0], self.x_range[1], 1],
                y_range=[self.y_range[0], self.y_range[1], 1],
                x_length=10,
                y_length=8,
                axis_config={
                    "include_numbers": False,
                    "include_ticks": False
                }
            ).scale(0.8)

            axes.center()
            
            x_label = axes.get_x_axis_label("x").scale(0.8)
            y_label = axes.get_y_axis_label("y").scale(0.8)

            try:
                # Prepare a clean display expression for LaTeX
                display_expr = "y=" + self._prepare_display_text(self.display_text)
                
                if self.logger:
                    self.logger.debug(f"LaTeX display expression: {display_expr}")
                    
                # Create the actual function for evaluation using SymPy
                import sympy as sp
                from sympy.abc import x
                from sympy import sympify, lambdify
                
                # Import common mathematical functions to be available during evaluation
                from sympy import sin, cos, tan, exp, log, sqrt
                
                # Clean up the expression for SymPy
                expr_str = str(self.func_str).strip()
                
                if self.logger:
                    self.logger.debug(f"Final expression for evaluation: '{expr_str}'")
                
                # Try to parse the expression directly without tokens
                try:
                    # First attempt: direct parsing
                    expr = sympify(expr_str)
                    f = lambdify(x, expr, "numpy")
                    
                    if self.logger:
                        self.logger.debug(f"Successfully parsed expression: {expr}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to parse expression '{expr_str}': {e}")
                    # Fall back to a simple linear function
                    expr = x
                    f = lambdify(x, expr, "numpy")
                
                # Create a denser sampling for smooth plotting
                num_samples = 2000  # Higher number for smoother curves
                x_min, x_max = self.x_range
                step = (x_max - x_min) / num_samples
                
                points = []
                valid_points = 0
                
                for i in range(num_samples + 1):
                    x_val = x_min + i * step
                    try:
                        y_val = float(f(x_val))
                        if self.y_range[0] <= y_val <= self.y_range[1]:
                            points.append((x_val, y_val, 0))
                            valid_points += 1
                    except Exception as e:
                        if self.logger:
                            self.logger.debug(f"Evaluation error at x={x_val}: {e}")
                
                if self.logger:
                    self.logger.debug(f"Generated {valid_points} valid points out of {num_samples + 1} total points")
                
                if valid_points == 0:
                    if self.logger:
                        self.logger.warning("No valid points found. Creating default linear function.")
                    # If no valid points are found, create a simple linear function
                    expr = x
                    f = lambdify(x, expr, "numpy")
                    points = []
                    for i in range(num_samples + 1):
                        x_val = x_min + i * step
                        try:
                            y_val = float(f(x_val))
                            if self.y_range[0] <= y_val <= self.y_range[1]:
                                points.append((x_val, y_val, 0))
                        except Exception as e:
                            if self.logger:
                                self.logger.debug(f"Evaluation error at x={x_val}: {e}")
                
                # Create the graph
                graph = axes.plot_line_graph(
                    x_values=[p[0] for p in points],
                    y_values=[p[1] for p in points],
                    line_color=BLUE,
                    add_vertex_dots=False
                )
                
                # Create label and position it in the top-right corner
                func_tex = MathTex(display_expr).scale(0.8)
                
                # Position the equation in the top-right corner
                func_tex.to_corner(UR, buff=0.5)
                
                # Add all elements to the scene
                self.add(axes, x_label, y_label, graph, func_tex)
                
                # Animate the drawing of the graph
                self.play(Create(axes), Create(x_label), Create(y_label))
                self.play(Write(func_tex))
                self.play(Create(graph))
                self.wait(1)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in FunctionScene: {e}")
                # Create fallback scene
                text = Text("Error visualizing function").scale(0.8)
                self.add(text)
                self.play(Write(text))
                self.wait(1)

        def _prepare_display_text(self, text):
            """Prepare text for LaTeX display by formatting mathematical functions and symbols.
            
            Args:
                text (str): The text to prepare for LaTeX display.
                
            Returns:
                str: The formatted text suitable for LaTeX display.
            """
            if text is None:
                return ""
            
            # Convert to string if not already
            text = str(text)
            
            # Replace ** with ^ for LaTeX
            text = text.replace("**", "^")
            
            # Replace * with \cdot for LaTeX
            text = text.replace("*", "\\cdot ")
            
            # Format common mathematical functions for LaTeX
            for func in ["sin", "cos", "tan", "exp", "log", "sqrt", "ln", "arcsin", "arccos", "arctan"]:
                # Replace func(x) with \func{x}
                text = re.sub(r'{}(\([^)]*\))'.format(func), r'\\{}\1'.format(func), text)
            
            # Additional replacements for LaTeX display
            text = text.replace("pi", "\\pi ")
            text = text.replace("sqrt", "\\sqrt")
            
            # Remove any remaining token patterns that might have been left
            text = re.sub(r'sympy_func_\d+', '', text)
            text = re.sub(r'SYMPY_FUNC_\d+_', '', text)
            text = re.sub(r'FUNCTOKEN\d+', '', text)
            
            return text

    def visualize_function(self, func_str: str, x_range: tuple = (-10, 10), y_range: tuple = (-5, 5)):
        """
        Create a visualization of a mathematical function.
        Returns the path to the generated video file.
        """
        try:
            self.logger.info(f"Visualizing function: {func_str}")
            
            # Clean up the function string
            func_str = func_str.strip()
            if not func_str:
                raise ValueError("Empty function string")
            
            # Note: We assume func_str is already properly formatted from _format_expression_for_sympy
            
            # The media_dir is already configured by the VisualizationWindow before calling us
            # We'll just make sure the video_dir is available
            video_dir = os.path.join(config.media_dir, "videos", "1080p60")
            os.makedirs(video_dir, exist_ok=True)
            
            # Verify that we're using the correct directories
            self.logger.debug(f"Using media_dir: {config.media_dir}")
            self.logger.debug(f"Using video_dir: {video_dir}")
            
            # Create and render the scene
            scene = self.FunctionScene(
                func_str,
                x_range=x_range,
                y_range=y_range,
                display_text=func_str,
                logger=self.logger
            )
            scene.render()
            
            # Return the path to the generated video
            video_path = os.path.join(video_dir, "FunctionScene.mp4")
            if os.path.exists(video_path):
                self.logger.info(f"Video file created at: {video_path}")
                return os.path.abspath(video_path)
            else:
                self.logger.error("Video file not created")
                return None
            
        except Exception as e:
            self.logger.error(f"Error in visualization: {str(e)}")
            return None