#!/usr/bin/env python3
import matlab.engine
import logging
from matlab_interface.evaluate_expression import EvaluateExpression

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_partial_derivatives')

def test_partial_derivatives():
    # Start MATLAB engine
    logger.info("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    
    # Create evaluator
    logger.info("Creating expression evaluator...")
    evaluator = EvaluateExpression(eng)
    
    # Test various partial derivative notations
    test_cases = [
        ("partial/dx sin(x)*cos(y)", "Standard partial derivative notation"),
        ("partial/dy sin(x)*cos(y)", "Partial derivative with respect to y"),
        ("partial2/dx2 sin(x)*cos(y)", "Second order partial derivative"),
        ("partial2/dxdy sin(x)*cos(y)", "Mixed partial derivative"),
        ("pdx sin(x)*cos(y)", "Shorthand pd notation"),
        ("pdy sin(x)*cos(y)", "Shorthand pd notation for y"),
        ("pd2x sin(x)*cos(y)", "Second order pd notation"),
        ("pd2xy sin(x)*cos(y)", "Mixed pd notation"),
        ("pd3xyz x*y*z", "Triple mixed partial derivative")
    ]
    
    for expr, desc in test_cases:
        logger.info(f"Testing {desc}: {expr}")
        try:
            # Manually preprocess the expression first
            preprocessed_expr = evaluator._preprocess_expression(expr)
            logger.info(f"Preprocessed expression: {preprocessed_expr}")
            
            # Now evaluate the preprocessed expression
            result = evaluator.evaluate(preprocessed_expr)
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Error: {e}")
    
    # Shut down MATLAB engine
    logger.info("Shutting down MATLAB engine...")
    eng.quit()
    logger.info("Test complete.")

if __name__ == "__main__":
    test_partial_derivatives() 