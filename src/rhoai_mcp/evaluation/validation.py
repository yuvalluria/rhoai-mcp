"""Result validation for the evaluation harness.

This module provides the ResultValidator class for validating
tool results against expected outcomes.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from rhoai_mcp.evaluation.models import ExpectedResult, ValidationResult

logger = logging.getLogger(__name__)


class ResultValidator:
    """Validates tool results against expected outcomes.

    Supports multiple validation strategies including required fields,
    exact value matches, regex patterns, and custom validators.
    """

    def __init__(self) -> None:
        """Initialize the validator."""
        self._custom_validators: dict[str, Callable[[Any, ExpectedResult], tuple[bool, str]]] = {}

    def register_validator(
        self,
        name: str,
        func: Callable[[Any, ExpectedResult], tuple[bool, str]],
    ) -> None:
        """Register a custom validator function.

        Args:
            name: Name to identify the validator.
            func: Validator function that takes (result, expected) and returns
                  (passed, error_message).
        """
        self._custom_validators[name] = func
        logger.debug(f"Registered custom validator: {name}")

    def unregister_validator(self, name: str) -> None:
        """Unregister a custom validator.

        Args:
            name: Name of the validator to remove.
        """
        self._custom_validators.pop(name, None)

    def validate(self, result: Any, expected: ExpectedResult) -> ValidationResult:
        """Validate a result against expected outcome.

        Args:
            result: The actual result to validate.
            expected: The expected result specification.

        Returns:
            ValidationResult with pass/fail status and failure details.
        """
        failures: list[str] = []

        # Check required fields
        for field in expected.required_fields:
            if not self._has_field(result, field):
                failures.append(f"Missing required field: {field}")

        # Check exact field values
        for field, expected_value in expected.field_values.items():
            actual = self._get_field(result, field)
            if actual != expected_value:
                failures.append(f"Field '{field}': expected {expected_value!r}, got {actual!r}")

        # Check field patterns (regex)
        for field, pattern in expected.field_patterns.items():
            actual = self._get_field(result, field)
            if actual is None:
                failures.append(f"Field '{field}' not found for pattern matching")
            elif not re.match(pattern, str(actual)):
                failures.append(
                    f"Field '{field}' value '{actual}' does not match pattern '{pattern}'"
                )

        # Run custom validator
        if expected.custom_validator:
            validator = self._custom_validators.get(expected.custom_validator)
            if validator:
                try:
                    passed, error_msg = validator(result, expected)
                    if not passed:
                        failures.append(
                            f"Custom validator '{expected.custom_validator}': {error_msg}"
                        )
                except Exception as e:
                    failures.append(f"Custom validator '{expected.custom_validator}' error: {e}")
            else:
                failures.append(f"Custom validator '{expected.custom_validator}' not registered")

        return ValidationResult(
            passed=len(failures) == 0,
            expected=expected,
            actual_result=result,
            failures=failures,
        )

    def validate_many(
        self,
        results: list[tuple[Any, ExpectedResult]],
    ) -> list[ValidationResult]:
        """Validate multiple results.

        Args:
            results: List of (result, expected) tuples.

        Returns:
            List of ValidationResults.
        """
        return [self.validate(result, expected) for result, expected in results]

    def _has_field(self, obj: Any, field: str) -> bool:
        """Check if an object has a field (supports nested fields with dot notation).

        Args:
            obj: Object to check.
            field: Field name (supports 'a.b.c' notation).

        Returns:
            True if the field exists.
        """
        parts = field.split(".")
        current = obj

        for part in parts:
            if current is None:
                return False

            if isinstance(current, dict):
                if part not in current:
                    return False
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return False

        return True

    def _get_field(self, obj: Any, field: str) -> Any:
        """Get a field value from an object (supports nested fields with dot notation).

        Args:
            obj: Object to get value from.
            field: Field name (supports 'a.b.c' notation).

        Returns:
            The field value, or None if not found.
        """
        parts = field.split(".")
        current = obj

        for part in parts:
            if current is None:
                return None

            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

        return current


# Built-in custom validators


def validate_not_empty(result: Any, expected: ExpectedResult) -> tuple[bool, str]:  # noqa: ARG001
    """Validate that a result is not empty.

    Works with strings, lists, dicts, and None.
    """
    if result is None:
        return False, "Result is None"
    if isinstance(result, str) and not result.strip():
        return False, "Result is empty string"
    if isinstance(result, list | dict) and not result:
        return False, "Result is empty collection"
    return True, ""


def validate_is_dict(result: Any, expected: ExpectedResult) -> tuple[bool, str]:  # noqa: ARG001
    """Validate that a result is a dictionary."""
    if not isinstance(result, dict):
        return False, f"Expected dict, got {type(result).__name__}"
    return True, ""


def validate_is_list(result: Any, expected: ExpectedResult) -> tuple[bool, str]:  # noqa: ARG001
    """Validate that a result is a list."""
    if not isinstance(result, list):
        return False, f"Expected list, got {type(result).__name__}"
    return True, ""


def validate_no_error(result: Any, expected: ExpectedResult) -> tuple[bool, str]:  # noqa: ARG001
    """Validate that a result doesn't contain an error field."""
    if isinstance(result, dict) and "error" in result:
        return False, f"Result contains error: {result['error']}"
    return True, ""


def validate_success_field(result: Any, expected: ExpectedResult) -> tuple[bool, str]:  # noqa: ARG001
    """Validate that a result has a truthy 'success' field."""
    if isinstance(result, dict):
        if "success" not in result:
            return False, "Result missing 'success' field"
        if not result["success"]:
            return False, f"Result 'success' is falsy: {result.get('success')}"
    return True, ""


def create_default_validator() -> ResultValidator:
    """Create a ResultValidator with built-in validators registered.

    Returns:
        A ResultValidator with common validators pre-registered.
    """
    validator = ResultValidator()
    validator.register_validator("not_empty", validate_not_empty)
    validator.register_validator("is_dict", validate_is_dict)
    validator.register_validator("is_list", validate_is_list)
    validator.register_validator("no_error", validate_no_error)
    validator.register_validator("success_field", validate_success_field)
    return validator
