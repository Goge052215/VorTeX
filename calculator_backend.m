function result = calculator_backend(expression)
    try
        % Evaluate the expression safely
        result = eval(expression);
    catch ME
        error(['Error in MATLAB backend: ' ME.message]);
    end
end