class LMKGException(Exception):
    """Base class for all lmkg exceptions."""

    message = "An exception has occurred"

    def __init__(self, message=None):
        super().__init__(message)


class MalformedQueryException(LMKGException):
    """Raised when a SPARQL query is malformed."""
    message = "Attempted to run a malformed graph query."
