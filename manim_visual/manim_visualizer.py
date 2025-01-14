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
        def __init__(self, func_str, x_range=(-10, 10), y_range=(-5, 5), display_text=None, logger=None):
            super().__init__()
            self.func_str = func_str
            self.display_text = display_text or func_str
            self.x_range = x_range
            self.y_range = y_range
            self.logger = logger

        def construct(self):
            try:
                if isinstance(self.func_str, tuple):
                    self.func_str = self.func_str[0]

                axes = Axes(
                    x_range=[self.x_range[0], self.x_range[1], (self.x_range[1] - self.x_range[0]) / 10],
                    y_range=[self.y_range[0], self.y_range[1], (self.y_range[1] - self.y_range[0]) / 10],
                    tips=True,
                    y_length=8,
                    axis_config={"include_numbers": False}
                ).scale(0.8)

                axes.center()
                
                x_label = axes.get_x_axis_label("x").scale(0.8)
                y_label = axes.get_y_axis_label("y").scale(0.8)

                try:
                    display_expr = self.func_str
                    display_expr = "y=" + display_expr
                    
                    # Add specific handling for e^x notation
                    display_expr = re.sub(r'e\^x', r'e^{x}', display_expr)
                    display_expr = re.sub(r'exp\(x\)', r'e^{x}', display_expr)
                    
                    display_expr = re.sub(r'(\d+)\*([a-zA-Z])', r'\1\2', display_expr)
                    
                    display_expr = re.sub(r'log(\d+)\(([^)]+)\)', r'\\log_{\1} \2', display_expr)
                    display_expr = re.sub(r'log\(([^)]+)\)', r'\\ln \1', display_expr)
                    display_expr = re.sub(r'ln\(([^)]+)\)', r'\\ln \1', display_expr)

                    trig_functions = {
                        'arcsin': 'sin^{-1}',
                        'arccos': 'cos^{-1}',
                        'arctan': 'tan^{-1}',
                        'arccot': 'cot^{-1}',
                        'arcsec': 'sec^{-1}',
                        'arccsc': 'csc^{-1}',
                        'sin': 'sin',
                        'cos': 'cos',
                        'tan': 'tan',
                        'cot': 'cot',
                        'sec': 'sec',
                        'csc': 'csc'
                    }
                    
                    hyperbolic_functions = {
                        'arcsinh': 'sinh^{-1}',
                        'arccosh': 'cosh^{-1}',
                        'arctanh': 'tanh^{-1}',
                        'arccoth': 'coth^{-1}',
                        'sinh': 'sinh',
                        'cosh': 'cosh',
                        'tanh': 'tanh',
                        'coth': 'coth'
                    }
                    
                    for func, latex_func in trig_functions.items():
                        display_expr = re.sub(
                            rf'{func}\(([^)]+)\)',
                            rf'\\{latex_func} \1',
                            display_expr
                        )
                    
                    for func, latex_func in hyperbolic_functions.items():
                        display_expr = re.sub(
                            rf'{func}\(([^)]+)\)',
                            rf'\\{latex_func} \1',
                            display_expr
                        )
                    
                    # Convert the expression for computation
                    expr_str = re.sub(r'\\log_(\d+)\s+([^)]+)', r'log(\2)/log(\1)', display_expr)
                    expr_str = re.sub(r'\\ln\s+([^)]+)', r'log(\1)', expr_str)
                    
                    # Handle equation case (contains '=')
                    if '=' in expr_str:
                        left_side, right_side = expr_str.split('=')
                        expr_str = f"({left_side})-({right_side})"
                    else:
                        expr_str = self.func_str

                    expr_str = re.sub(r'e\^(\([^)]+\)|\w+)', lambda m: f'exp{m.group(1)}', expr_str)
                    expr_str = re.sub(r'log(\d+)\(([^)]+)\)', lambda m: f'log({m.group(2)})/log({m.group(1)})', expr_str)
                    
                    # Update expression string handling for e^x
                    expr_str = self.func_str
                    expr_str = re.sub(r'e\^x', r'exp(x)', expr_str)
                    
                    # Create a safe lambda function using numpy for evaluation
                    import numpy as np
                    x = Symbol('x')
                    expr = sympify(expr_str)
                    
                    def safe_eval(x_val):
                        try:
                            result = float(expr.subs('x', x_val))
                            if abs(result) > self.y_range[1] or np.isnan(result) or np.isinf(result):
                                return None
                            return result
                        except:
                            return None

                    x_vals = np.linspace(self.x_range[0], self.x_range[1], 2000)
                    y_vals = [safe_eval(x_val) for x_val in x_vals]
                    
                    valid_points = [(x, y) for x, y in zip(x_vals, y_vals) if y is not None]
                    if valid_points:
                        x_coords, y_coords = zip(*valid_points)
                        
                        graph = VMobject()
                        graph.set_points_smoothly([
                            axes.c2p(x, y) for x, y in zip(x_coords, y_coords)
                        ])
                        graph.set_color(YELLOW)

                        x_right = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * 0.7
                        try:
                            y_right = safe_eval(x_right)
                            if y_right is None:
                                y_right = self.y_range[1] * 0.7
                            y_offset = (self.y_range[1] - self.y_range[0]) * 0.1
                            y_label_pos = min(y_right + y_offset, self.y_range[1] * 0.8)
                        except:
                            y_label_pos = self.y_range[1] * 0.7
                        
                        label = MathTex(display_expr).scale(0.8)
                        label.next_to(graph, RIGHT, buff=0.5)
                        
                        self.add(axes, graph, label)

                        # Add all elements to the scene with animations
                        self.play(Create(axes), run_time=1)
                        self.play(Write(x_label), Write(y_label), run_time=0.5)
                        self.play(Create(graph), run_time=2)
                        self.play(Write(label), run_time=1)
                        
                        self.wait(2)
                    else:
                        raise ValueError("No valid points to plot")

                except Exception as e:
                    self.logger.error(f"Error creating graph: {e}")
                    error_text = Text(f"Error: {str(e)}", color=RED).scale(0.6)
                    self.add(error_text)
                    self.wait(2)

            except Exception as e:
                self.logger.error(f"Error in visualization: {e}")
                error_text = Text("Error visualizing function", color=RED).scale(0.6)
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
            func_str = re.sub(r'(\d+)\s*([a-zA-Z])', r'\1*\2', func_str)
            func_str = re.sub(r'(\d+)/(\d+)\s*([a-zA-Z])', r'(\1/\2)*\3', func_str)
            func_str = re.sub(r'\s+', '', func_str)
            
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
                func_str = f"{func_str}+0*x"
                
            self.logger.debug(f"Processed function string: {func_str}")
            
            safe_dict = {
                "x": np.linspace(-1, 1, 10),
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
            
            try:
                eval(func_str, {"__builtins__": {}}, safe_dict)
            except Exception as e:
                self.logger.error(f"Invalid function syntax: {e}")
                raise ValueError(f"Invalid function syntax: {e}")
            
            scene = self.FunctionScene(func_str, x_range, y_range, logger=self.logger)
            scene.render()
            
        except Exception as e:
            self.logger.error(f"Error visualizing function: {e}")
            raise