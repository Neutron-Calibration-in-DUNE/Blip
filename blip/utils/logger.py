"""
Logger class for all blip classes.
"""
import warnings
import logging
import platform
import traceback
import subprocess
import socket
import re
import uuid
import psutil
import os
import torch


class BlipError(Exception):
    """Custom default error for Blip"""
    def __init__(
        self,
        message="An error occurred"
    ):
        self.message = message
        super().__init__(self.message)


class EventError(Exception):
    """Custom default error for an event"""
    def __init__(
        self,
        message="An error occurred"
    ):
        self.message = message
        super().__init__(self.message)


logging_level = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

warning_list = {
    "deprecation": DeprecationWarning,
    "import": ImportWarning,
    "resource": ResourceWarning,
    "user": UserWarning,
}

error_list = {
    "attribute": AttributeError,
    "blip": BlipError,
    "event": EventError,
    "index": IndexError,
    "file": FileExistsError,
    "memory": MemoryError,
    "runtime": RuntimeError,
    "type": TypeError,
    "value": ValueError,
}


class disable_logging(object):
    """
    A context manager to disable logging temporarily.
    """

    def __init__(self, level=logging.ERROR):  # pragma: no cover
        """
        Initialize the context manager.
        """
        logging.disable(level=level)

    def __enter__(self):  # pragma: no cover
        """
        Enter the context manager.
        """
        return self

    def __exit__(self, type, value, traceback):  # pragma: no cover
        """
        Exit the context manager and enable logging.
        """
        logging.disable(level=logging.NOTSET)


class LoggingFormatter(logging.Formatter):
    """
    Formatting for Blip logging.

    Args:
        logging (_logging.Formatter_): The formatter
        to be edited.

    Returns:
        _logging.Formatter_: The edited formatter
    """
    console_extra = " [%(name)s]: %(message)s"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[1;34m"
    purple = "\x1b[1;35m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    FORMATS = {
        logging.DEBUG: "[" + grey + "%(levelname)s" + reset + "] [" + purple + "%(name)s" + reset + "]: %(message)s",
        logging.INFO: "[" + blue + "%(levelname)s" + reset + "] [" + purple + "%(name)s" + reset + "]: %(message)s",
        logging.WARNING: "[" + yellow + "%(levelname)s" + reset + "] [" + purple + "%(name)s" + reset + "]: %(message)s",
        logging.ERROR: "[" + red + "%(levelname)s" + reset + "] [" + purple + "%(name)s" + reset + "]: %(message)s",
        logging.CRITICAL: "[" + bold_red + "%(levelname)s" + reset + "] [" + purple + "%(name)s" + reset + "]: %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger:
    """
    Custom logging wrapper for Blip programs.
    The logger writes to three different streams:
        (1) console
        (2) file
        (3) debug
    """

    def __init__(
        self,
        meta: dict = {},
    ):
        """
        Initializer for the logger

        Args:
            name (str, optional): name for this logger (will appear
            after the log message type [ERROR] [name]). Defaults to "default".

            level (str, optional): _description_. Defaults to "debug".
            output (str, optional): whether to log to console, file
            or both. Defaults to "file".

            run_name (str, optional): name of the output file. Defaults to "log".
            file_mode (str, optional): whether to append, or rewrite
            log files for this run. Defaults to "a".

        Raises:
            ValueError: _description_
        """
        self.meta = meta

        self.local_log_dir = os.path.join(
            self.meta["experiment_directory"], "logs/"
        )
        if not os.path.isdir(self.local_log_dir):
            os.makedirs(self.local_log_dir)

        # use the name as the default output file name
        self.run_name = self.meta["run_name"]
        self.level = logging_level["debug"]
        self.file_mode = "a"

        # create logger
        self.logger = logging.getLogger(self.run_name)
        self.debug_logger = logging.getLogger(self.run_name + "_debug")

        # set level
        self.logger.setLevel(self.level)
        self.debug_logger.setLevel(self.level)

        # set format
        self.dateformat = "%H:%M:%S"

        self.console_formatter = LoggingFormatter()
        self.file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", self.dateformat
        )

        self.debug = logging.FileHandler(
            self.local_log_dir + self.run_name + ".debug", mode="a"
        )
        self.debug.setLevel(self.level)

        # create handler
        self.console = logging.StreamHandler()
        self.console.setLevel(self.level)
        self.console.setFormatter(self.console_formatter)
        self.logger.addHandler(self.console)
        self.file = logging.FileHandler(
            self.local_log_dir + self.run_name + ".log", mode="a"
        )
        self.file.setLevel(logging.DEBUG)
        self.file.setFormatter(self.file_formatter)
        self.logger.addHandler(self.file)
        self.debug.setFormatter(self.file)
        self.debug_logger.addHandler(self.debug)
        self.logger.propagate = False

    def info(
        self,
        message: str,
    ):
        """_summary_

        Args:
            message (str): _description_

        Returns:
            _type_: _description_
        """
        """Output to the standard logger "info" """
        return self.logger.info(message)

    def debug(
        self,
        message: str,
    ):
        """_summary_

        Args:
            message (str): _description_

        Returns:
            _type_: _description_
        """
        """Output to the standard logger "debug" """
        return self.debug_logger.debug(message)

    def warn(
        self,
        message: str,
    ):
        """_summary_

        Args:
            message (str): _description_

        Returns:
            _type_: _description_
        """
        """Output to the standard logger "warning" """
        return self.logger.warning(message)

    def warning(
        self,
        message: str,
        warning_type: str = "user",
    ):
        """_summary_

        Args:
            message (str): _description_
            warning_type (str, optional): _description_. Defaults to "user".

        Returns:
            _type_: _description_
        """
        """Output to the standard logger "warning" """
        formatted_lines = traceback.format_stack()[-2]
        if warning_type not in warning_list.keys():
            warning_type = "user"
        self.logger.warning(message)
        warnings.warn(
            f"traceback: {formatted_lines}\nerror: {message}",
            warning_list[warning_type],
        )
        return

    def error(
        self,
        message: str,
        error_type: str = "blip",
    ):
        """_summary_

        Args:
            message (str): _description_
            error_type (str, optional): _description_. Defaults to "value".

        Raises:
            error_list: _description_
        """
        """Output to the standard logger "error" """
        formatted_traceback = ''.join(traceback.format_stack())
        if error_type not in error_list.keys():
            error_type = "blip"
        log_message = f"Traceback: \n{formatted_traceback}\nError: {message}"
        self.logger.error(message)
        raise error_list[error_type](log_message)

    def critical(
        self,
        message: str
    ):
        """
        """
        return self.logger.critical(message)

    def get_system_info(
        self
    ) -> dict:
        """
        Attempt to get system info using various
        python packages.  If this fails, return
        an empty dictionary.

        Returns:
            _dict_: dictionary containing system info.
        """
        info = {
            "run_name": self.run_name,
            "run_number": self.meta["run_num"],
            "world_size": self.meta["world_size"],
            "world_rank": self.meta["world_rank"],
            "local_rank": self.meta["local_rank"],
            "num_data_shards": self.meta["num_data_shards"],
        }
        try:
            info["platform"] = platform.system()
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        try:
            info["platform-release"] = platform.release()
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        try:
            info["platform-version"] = platform.version()
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        try:
            info["architecture"] = platform.machine()
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        try:
            info["hostname"] = socket.gethostname()
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        try:
            info["ip-address"] = socket.gethostbyname(socket.gethostname())
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        try:
            info["mac-address"] = ":".join(re.findall("..", "%012x" % uuid.getnode()))
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        try:
            info["processor"] = platform.processor()
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        try:
            info["physical_cores"] = psutil.cpu_count(logical=False)
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        try:
            info["logical_cores"] = psutil.cpu_count(logical=True)
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        try:
            info["local_scratch"] = str(round(psutil.disk_usage(f"{os.environ['LOCAL_SCRATCH']}").free / (1024.0**3))) + " GB"
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        try:
            info["RAM"] = str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB"
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        try:
            info["git_branch"] = str(subprocess.check_output(['git', 'branch']).strip())
        except Exception as e:
            self.logger.warning(f"Unable to obtain git_branch: {e}")
        try:
            info["git_hash"] = str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())
        except Exception as e:
            self.logger.warning(f"Unable to obtain git_hash: {e}")
        try:
            info["torch_version"] = str(torch.__version__)
        except Exception as e:
            self.logger.warning(f"Unable to obtain torch_version: {e}")
        return info
