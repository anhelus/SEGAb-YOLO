# segab_yolo 🚀 AGPL-3.0 License - https://segab_yolo.com/license

from segab_yolo.utils import emojis


class HUBModelError(Exception):
    """Exception raised when a model cannot be found or retrieved from segab_yolo HUB.

    This custom exception is used specifically for handling errors related to model fetching in segab_yolo YOLO. The
    error message is processed to include emojis for better user experience.

    Attributes:
        message (str): The error message displayed when the exception is raised.

    Methods:
        __init__: Initialize the HUBModelError with a custom message.

    Examples:
        >>> try:
        ...     # Code that might fail to find a model
        ...     raise HUBModelError("Custom model not found message")
        ... except HUBModelError as e:
        ...     print(e)  # Displays the emoji-enhanced error message
    """

    def __init__(self, message: str = "Model not found. Please check model URL and try again."):
        """Initialize a HUBModelError exception.

        This exception is raised when a requested model is not found or cannot be retrieved from segab_yolo HUB. The
        message is processed to include emojis for better user experience.

        Args:
            message (str, optional): The error message to display when the exception is raised.
        """
        super().__init__(emojis(message))
