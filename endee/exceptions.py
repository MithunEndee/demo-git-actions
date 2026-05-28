"""
Exceptions Module for Endee Vector Database Client

This module defines custom exception classes for handling various error
conditions that can occur when interacting with the Endee API. Each exception
type corresponds to specific HTTP status codes and error scenarios.
"""

import orjson


class EndeeException(Exception):
    """
    Base class for all Endee-related exceptions.

    All custom exceptions in the Endee client inherit from this class,
    allowing for easy exception handling with a single catch block.

    Attributes:
        message (str): Human-readable error message
    """

    def __init__(self, message="An error occurred in Endee"):
        """
        Initialize the EndeeException.

        Args:
            message: Error message describing what went wrong
        """
        self.message = message
        super().__init__(message)

    def __str__(self):
        """
        String representation of the exception.

        Returns:
            str: The error message
        """
        return self.message


class APIException(EndeeException):
    """
    Generic API exception.

    Raised when an API call returns an error that doesn't fit other specific
    exception types. Typically corresponds to HTTP 400 (Bad Request) errors.

    Example:
        Invalid parameters, malformed requests, or validation errors.
    """

    def __init__(self, message):
        """
        Initialize the APIException.

        Args:
            message: Description of the API error
        """
        self.message = message
        super().__init__(f"API Error: {message}")


class ServerException(EndeeException):
    """
    Server error exception.

    Raised when the server encounters an internal error. Corresponds to
    HTTP 5xx status codes (500, 502, 503, 504).

    Example:
        Server overload, service unavailable, or internal server errors.
    """

    def __init__(self, message):
        """
        Initialize the ServerException.

        Args:
            message: Description of the server error
        """
        self.message = message
        super().__init__(f"Server Busy: {message}")


class ForbiddenException(EndeeException):
    """
    Permission denied exception.

    Raised when the user is not allowed to perform the requested operation.
    Corresponds to HTTP 403 (Forbidden) status code.

    Example:
        Attempting to delete another user's index, or performing admin
        operations without root privileges.
    """

    def __init__(self, message):
        """
        Initialize the ForbiddenException.

        Args:
            message: Description of why the operation is forbidden
        """
        self.message = message
        super().__init__(f"Forbidden: {message}")


class ConflictException(EndeeException):
    """
    Resource conflict exception.

    Raised when attempting to create a resource that already exists.
    Corresponds to HTTP 409 (Conflict) status code.

    Example:
        Attempting to create an index with a name that's already in use.
    """

    def __init__(self, message):
        """
        Initialize the ConflictException.

        Args:
            message: Description of the resource conflict
        """
        self.message = message
        super().__init__(f"Conflict: {message}")


class NotFoundException(EndeeException):
    """
    Resource not found exception.

    Raised when the requested resource does not exist. Corresponds to
    HTTP 404 (Not Found) status code.

    Example:
        Attempting to query an index that doesn't exist, or retrieving
        a vector with an ID that's not in the index.
    """

    def __init__(self, message):
        """
        Initialize the NotFoundException.

        Args:
            message: Description of what resource was not found
        """
        self.message = message
        super().__init__(f"Resource Not Found: {message}")


class AuthenticationException(EndeeException):
    """
    Authentication failure exception.

    Raised when the authentication token is invalid, expired, or missing.
    Corresponds to HTTP 401 (Unauthorized) status code.

    Example:
        Using an invalid or expired token, or making authenticated requests
        without providing credentials.
    """

    def __init__(self, message):
        """
        Initialize the AuthenticationException.

        Args:
            message: Description of the authentication error
        """
        self.message = message
        super().__init__(f"Authentication Error: {message}")


class SubscriptionException(EndeeException):
    """
    Subscription or payment required exception.

    Raised when the operation requires a paid subscription or when the user
    has exceeded their tier limits. Corresponds to HTTP 402 (Payment Required)
    status code.

    Example:
        Attempting to create more indexes than allowed on the Free tier,
        or accessing Pro features without a Pro subscription.
    """

    def __init__(self, message):
        """
        Initialize the SubscriptionException.

        Args:
            message: Description of the subscription issue
        """
        self.message = message
        super().__init__(f"Subscription Error: {message}")


def raise_exception(code: int, text: str = None):
    """
    Raise an appropriate exception based on the HTTP status code.

    Maps HTTP status codes to corresponding exception types and extracts
    the error message from the response text. If the response is JSON,
    extracts the "error" field; otherwise uses the raw text.

    Args:
        code: HTTP status code from the API response
        text: Response body text (may be JSON or plain text)

    Raises:
        APIException: For HTTP 400 (Bad Request) or unknown errors
        AuthenticationException: For HTTP 401 (Unauthorized)
        SubscriptionException: For HTTP 402 (Payment Required)
        ForbiddenException: For HTTP 403 (Forbidden)
        NotFoundException: For HTTP 404 (Not Found)
        ConflictException: For HTTP 409 (Conflict)
        ServerException: For HTTP 5xx (Server Errors)

    Example:
        >>> try:
        ...     response = client.get("/index/nonexistent")
        ...     if response.status_code != 200:
        ...         raise_exception(response.status_code, response.text)
        ... except NotFoundException as e:
        ...     print(f"Index not found: {e}")
    """
    # Try to parse JSON error message
    message = None
    try:
        message = orjson.loads(text).get("error", "Unknown error")
    except (orjson.JSONDecodeError, TypeError, AttributeError):
        # Fall back to raw text or default message
        message = text or "Unknown error"

    # Map status codes to exception types
    if code == 400:
        raise APIException(message)
    elif code == 401:
        raise AuthenticationException(message)
    elif code == 402:
        raise SubscriptionException(message)
    elif code == 403:
        raise ForbiddenException(message)
    elif code == 404:
        raise NotFoundException(message)
    elif code == 409:
        raise ConflictException(message)
    elif code >= 500:
        # Generic server error message for all 5xx codes
        message = "Server is busy. Please try again in sometime"
        raise ServerException(message)
    else:
        # Fallback for unrecognized status codes
        message = "Unknown Error. Please try again in sometime"
        raise APIException(message)
