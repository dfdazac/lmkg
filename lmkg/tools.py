from datetime import datetime


def current_time():
    """Get the current local time as a string."""
    return str(datetime.now())


def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b


tools_dict = {"current_time": current_time,
         "multiply": multiply}
