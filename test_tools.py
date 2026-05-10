import pytest
from assignment import calculator, code_execution

# ==========================================
# CALCULATOR TOOL TESTS
# ==========================================

def test_calculator_valid_expression():
    """Test that the calculator correctly evaluates valid math."""
    state = {"tool_input": "2 * 3 * (4 / 2)"}
    result = calculator(state)
    
    # numexpr returns floats, so it will be "12.0"
    assert "calculation_result" in result
    assert result["calculation_result"] == "12.0"

def test_calculator_invalid_expression():
    """Test that the calculator safely catches math errors instead of crashing."""
    state = {"tool_input": "2 / 0"}
    result = calculator(state)
    
    assert "calculation_result" in result
    assert "Error" in result["calculation_result"]

def test_calculator_syntax_error():
    """Test that the calculator catches malformed strings."""
    state = {"tool_input": "2 * * 3"}
    result = calculator(state)
    
    assert "calculation_result" in result
    assert "Error" in result["calculation_result"]

# ==========================================
# CODE EXECUTION TOOL TESTS
# ==========================================

def test_code_execution_valid():
    """Test that the REPL successfully runs basic Python code."""
    state = {"tool_input": "print('Hello from the sandbox!')"}
    result = code_execution(state)
    
    assert "code_execution_result" in result
    assert "Hello from the sandbox!" in result["code_execution_result"]

def test_code_execution_error():
    """Test that the REPL returns the stack trace gracefully on bad code."""
    state = {"tool_input": "print(undefined_variable)"}
    result = code_execution(state)
    
    assert "code_execution_result" in result
    assert "NameError" in result["code_execution_result"]