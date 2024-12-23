from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QFont

def handle_exception(message, exception):
    """
    Logs the exception and displays a critical message box to the user.
    """
    print(f"{message}: {exception}")
    if exception:
        error_message = f"{message}: {str(exception)}"
    else:
        error_message = message
    QMessageBox.critical(None, "Error", error_message)
