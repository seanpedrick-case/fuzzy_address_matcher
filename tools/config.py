import os
import re
import socket
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import List

import bleach
from tools.secure_path_utils import (
    secure_path_join,
)

today_rev = datetime.now().strftime("%Y%m%d")
HOST_NAME = socket.gethostname()


def _get_env_list(env_var_name: str) -> List[str]:
    """Parses a comma-separated environment variable into a list of strings."""
    value = env_var_name[1:-1].strip().replace('"', "").replace("'", "")
    if not value:
        return []
    # Split by comma and filter out any empty strings that might result from extra commas
    return [s.strip() for s in value.split(",") if s.strip()]


# Set or retrieve configuration variables for the redaction app


def convert_string_to_boolean(value: str) -> bool:
    """Convert string to boolean, handling various formats."""
    if isinstance(value, bool):
        return value
    elif value in ["True", "1", "true", "TRUE"]:
        return True
    elif value in ["False", "0", "false", "FALSE"]:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")


def ensure_folder_within_app_directory(
    folder_path: str, app_base_dir: str = None
) -> str:
    """
    Ensure that a folder path is within the app directory for security.

    This function validates that user-defined folder paths are contained within
    the app directory to prevent path traversal attacks and ensure data isolation.

    Args:
        folder_path: The folder path to validate and normalize
        app_base_dir: The base directory of the app (defaults to current working directory)

    Returns:
        A normalized folder path that is guaranteed to be within the app directory

    Raises:
        ValueError: If the path cannot be safely contained within the app directory
    """
    if not folder_path or not folder_path.strip():
        return folder_path

    # Get the app base directory (where the app is run from)
    if app_base_dir is None:
        app_base_dir = os.getcwd()

    app_base_dir = Path(app_base_dir).resolve()
    folder_path = folder_path.strip()

    # Preserve trailing separator preference
    has_trailing_sep = folder_path.endswith(("/", "\\"))

    # Handle special case for "TEMP" - this is handled separately in the code
    if folder_path == "TEMP":
        return folder_path

    # Handle absolute paths
    if os.path.isabs(folder_path):
        folder_path_resolved = Path(folder_path).resolve()
        # Check if the absolute path is within the app directory
        try:
            folder_path_resolved.relative_to(app_base_dir)
            # Path is already within app directory, return it normalized
            result = str(folder_path_resolved)
            if has_trailing_sep and not result.endswith(os.sep):
                result = result + os.sep
            return result
        except ValueError:
            # Path is outside app directory - this is a security issue
            # For system paths like /usr/share/tessdata, we'll allow them but log a warning
            # For other absolute paths outside app directory, we'll raise an error
            normalized_path = os.path.normpath(folder_path).lower()
            system_path_prefixes = [
                "/usr",
                "/opt",
                "/var",
                "/etc",
                "/tmp",
            ]
            if any(
                normalized_path.startswith(prefix) for prefix in system_path_prefixes
            ):
                # System paths are allowed but we log a warning
                print(
                    f"Warning: Using system path outside app directory: {folder_path}"
                )
                return folder_path
            else:
                raise ValueError(
                    f"Folder path '{folder_path}' is outside the app directory '{app_base_dir}'. "
                    f"For security, all user-defined folder paths must be within the app directory."
                )

    # Handle relative paths - ensure they're within app directory
    try:
        # Use secure_path_join to safely join and validate
        # This will prevent path traversal attacks (e.g., "../../etc/passwd")
        safe_path = secure_path_join(app_base_dir, folder_path)
        result = str(safe_path)
        if has_trailing_sep and not result.endswith(os.sep):
            result = result + os.sep
        return result
    except (PermissionError, ValueError) as e:
        # If path contains dangerous patterns, sanitize and try again
        # Extract just the folder name from the path to prevent traversal
        folder_name = os.path.basename(folder_path.rstrip("/\\"))
        if folder_name:
            safe_path = secure_path_join(app_base_dir, folder_name)
            result = str(safe_path)
            if has_trailing_sep and not result.endswith(os.sep):
                result = result + os.sep
            print(
                f"Warning: Sanitized folder path '{folder_path}' to '{result}' for security"
            )
            return result
        else:
            raise ValueError(
                f"Cannot safely normalize folder path: {folder_path}"
            ) from e


def get_or_create_env_var(var_name: str, default_value: str, print_val: bool = False):
    """
    Get an environmental variable, and set it to a default value if it doesn't exist
    """
    # Get the environment variable if it exists
    value = os.environ.get(var_name)

    # If it doesn't exist, set the environment variable to the default value
    if value is None:
        os.environ[var_name] = default_value
        value = default_value

    if print_val is True:
        print(f"The value of {var_name} is {value}")

    return value


def add_folder_to_path(folder_path: str):
    """
    Check if a folder exists on your system. If so, get the absolute path and then add it to the system Path variable if it doesn't already exist. Function is only relevant for locally-created executable files based on this app (when using pyinstaller it creates a _internal folder that contains tesseract and poppler. These need to be added to the system path to enable the app to run)
    """

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # print(folder_path, "folder exists.")

        # Resolve relative path to absolute path
        absolute_path = os.path.abspath(folder_path)

        current_path = os.environ["PATH"]
        if absolute_path not in current_path.split(os.pathsep):
            full_path_extension = absolute_path + os.pathsep + current_path
            os.environ["PATH"] = full_path_extension
            # print(f"Updated PATH with: ", full_path_extension)
        else:
            pass
            # print(f"Directory {folder_path} already exists in PATH.")
    else:
        print(f"Folder not found at {folder_path} - not added to PATH")


def validate_safe_url(url_candidate: str, allowed_domains: list = None) -> str:
    """
    Validate and return a safe URL with enhanced security checks.
    """
    if allowed_domains is None:
        allowed_domains = [
            "seanpedrick-case.github.io",
            "github.io",
            "github.com",
            "sharepoint.com",
        ]

    try:
        parsed = urllib.parse.urlparse(url_candidate)

        # Basic structure validation
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL structure")

        # Security checks
        if parsed.scheme not in ["https"]:  # Only allow HTTPS
            raise ValueError("Only HTTPS URLs are allowed for security")

        # Domain validation
        domain = parsed.netloc.lower()
        if not any(domain.endswith(allowed) for allowed in allowed_domains):
            raise ValueError(f"Domain not in allowed list: {domain}")

        # Additional security checks
        if any(
            suspicious in domain for suspicious in ["..", "//", "javascript:", "data:"]
        ):
            raise ValueError("Suspicious URL patterns detected")

        # Path validation (prevent path traversal)
        if ".." in parsed.path or "//" in parsed.path:
            raise ValueError("Path traversal attempts detected")

        return url_candidate

    except Exception as e:
        print(f"URL validation failed: {e}")
        return "https://seanpedrick-case.github.io/doc_redaction"  # Safe fallback


def sanitize_markdown_text(text: str) -> str:
    """
    Sanitize markdown text by removing dangerous HTML/scripts while preserving
    safe markdown syntax.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove dangerous HTML tags and scripts using bleach
    # Define allowed tags for markdown (customize as needed)
    allowed_tags = [
        "a",
        "b",
        "strong",
        "em",
        "i",
        "u",
        "code",
        "pre",
        "blockquote",
        "ul",
        "ol",
        "li",
        "p",
        "br",
        "hr",
    ]
    allowed_attributes = {"a": ["href", "title", "rel"]}
    # Clean the text to strip (remove) any tags not in allowed_tags, and remove all script/iframe/etc.
    text = bleach.clean(
        text, tags=allowed_tags, attributes=allowed_attributes, strip=True
    )

    # Remove iframe, object, embed tags (should already be stripped, but keep for redundancy)
    text = re.sub(
        r"<(iframe|object|embed)[^>]*>.*?</\1>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove event handlers (onclick, onerror, etc.)
    text = re.sub(r'\s*on\w+\s*=\s*["\'][^"\']*["\']', "", text, flags=re.IGNORECASE)

    # Remove javascript: and data: URLs from markdown links
    text = re.sub(
        r"\[([^\]]+)\]\(javascript:[^\)]+\)", r"[\1]", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\[([^\]]+)\]\(data:[^\)]+\)", r"[\1]", text, flags=re.IGNORECASE)

    # Remove dangerous HTML attributes
    text = re.sub(
        r'\s*(style|onerror|onload|onclick)\s*=\s*["\'][^"\']*["\']',
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


S3_BUCKET_NAME = get_or_create_env_var("S3_BUCKET_NAME", "")

USE_POSTCODE_BLOCKER = convert_string_to_boolean(
    get_or_create_env_var("USE_POSTCODE_BLOCKER", "True")
)

MAX_PARALLEL_WORKERS = int(get_or_create_env_var("MAX_PARALLEL_WORKERS", "4"))
RUN_BATCHES_IN_PARALLEL = convert_string_to_boolean(
    get_or_create_env_var("RUN_BATCHES_IN_PARALLEL", "True")
)
