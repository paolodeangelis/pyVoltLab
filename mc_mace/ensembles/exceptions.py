class EnsembleError(Exception):
    """
    Base exception for MC ensemble errors.

    Attributes:
        message (str): The error message describing the exception.
    """

    def __init__(self, message: str = "MC ensemble general exception") -> None:
        """
        Initialize an `EnsembleError`.

        Args:
            message (str): Description of the error (default: "MC ensemble general exception").
        """
        self.message = message
        super().__init__(self.message)


class InvalidEnsembleAttemptType(EnsembleError):
    """
    Exception for invalid ensemble move types.

    Attributes:
        move_type (str): The invalid move type that caused the exception.
        message (str): The error message describing the issue.
    """

    def __init__(self, move_type: str, message: str = "Invalid move type selected") -> None:
        """
        Initialize an `InvalidEnsembleAttemptType`.

        Args:
            move_type (str): The invalid move type that triggered the exception.
            message (str): Description of the error (default: "Invalid move type selected").
        """
        self.move_type = move_type
        self.message = f"{message}: {move_type}"
        super().__init__(self.message)
