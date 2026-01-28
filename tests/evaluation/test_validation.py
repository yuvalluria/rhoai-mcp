"""Tests for result validation."""

import pytest

from rhoai_mcp.evaluation.models import ExpectedResult
from rhoai_mcp.evaluation.validation import (
    ResultValidator,
    create_default_validator,
    validate_is_dict,
    validate_is_list,
    validate_no_error,
    validate_not_empty,
    validate_success_field,
)


class TestResultValidator:
    """Test ResultValidator class."""

    def test_validate_required_fields_pass(self, validator) -> None:
        """Test validation passes when required fields exist."""
        expected = ExpectedResult(
            tool_name="test",
            required_fields=["name", "status"],
        )
        result = {"name": "test-item", "status": "ok", "extra": "data"}

        validation = validator.validate(result, expected)

        assert validation.passed is True
        assert validation.failures == []

    def test_validate_required_fields_fail(self, validator) -> None:
        """Test validation fails when required field is missing."""
        expected = ExpectedResult(
            tool_name="test",
            required_fields=["name", "status"],
        )
        result = {"name": "test-item"}  # Missing status

        validation = validator.validate(result, expected)

        assert validation.passed is False
        assert any("status" in f for f in validation.failures)

    def test_validate_nested_required_fields(self, validator) -> None:
        """Test validation of nested required fields."""
        expected = ExpectedResult(
            tool_name="test",
            required_fields=["data.name", "data.value"],
        )
        result = {"data": {"name": "test", "value": 123}}

        validation = validator.validate(result, expected)

        assert validation.passed is True

    def test_validate_field_values_pass(self, validator) -> None:
        """Test validation passes when field values match."""
        expected = ExpectedResult(
            tool_name="test",
            field_values={"status": "active", "count": 5},
        )
        result = {"status": "active", "count": 5}

        validation = validator.validate(result, expected)

        assert validation.passed is True

    def test_validate_field_values_fail(self, validator) -> None:
        """Test validation fails when field values don't match."""
        expected = ExpectedResult(
            tool_name="test",
            field_values={"status": "active"},
        )
        result = {"status": "inactive"}

        validation = validator.validate(result, expected)

        assert validation.passed is False
        assert any("status" in f for f in validation.failures)

    def test_validate_field_patterns_pass(self, validator) -> None:
        """Test validation passes when patterns match."""
        expected = ExpectedResult(
            tool_name="test",
            field_patterns={"name": r"^test-\d+$"},
        )
        result = {"name": "test-123"}

        validation = validator.validate(result, expected)

        assert validation.passed is True

    def test_validate_field_patterns_fail(self, validator) -> None:
        """Test validation fails when patterns don't match."""
        expected = ExpectedResult(
            tool_name="test",
            field_patterns={"name": r"^test-\d+$"},
        )
        result = {"name": "prod-123"}

        validation = validator.validate(result, expected)

        assert validation.passed is False
        assert any("pattern" in f for f in validation.failures)

    def test_validate_custom_validator_pass(self, validator) -> None:
        """Test validation with custom validator that passes."""
        validator.register_validator(
            "is_positive",
            lambda r, e: (r.get("value", 0) > 0, "Value must be positive"),
        )

        expected = ExpectedResult(
            tool_name="test",
            custom_validator="is_positive",
        )
        result = {"value": 10}

        validation = validator.validate(result, expected)

        assert validation.passed is True

    def test_validate_custom_validator_fail(self, validator) -> None:
        """Test validation with custom validator that fails."""
        validator.register_validator(
            "is_positive",
            lambda r, e: (r.get("value", 0) > 0, "Value must be positive"),
        )

        expected = ExpectedResult(
            tool_name="test",
            custom_validator="is_positive",
        )
        result = {"value": -5}

        validation = validator.validate(result, expected)

        assert validation.passed is False
        assert any("positive" in f for f in validation.failures)

    def test_validate_unregistered_custom_validator(self, validator) -> None:
        """Test validation fails when custom validator not registered."""
        expected = ExpectedResult(
            tool_name="test",
            custom_validator="nonexistent",
        )
        result = {"data": "test"}

        validation = validator.validate(result, expected)

        assert validation.passed is False
        assert any("not registered" in f for f in validation.failures)

    def test_validate_many(self, validator) -> None:
        """Test validating multiple results."""
        expected1 = ExpectedResult(tool_name="test", required_fields=["name"])
        expected2 = ExpectedResult(tool_name="test", required_fields=["status"])

        results = [
            ({"name": "test"}, expected1),
            ({"status": "ok"}, expected2),
        ]

        validations = validator.validate_many(results)

        assert len(validations) == 2
        assert all(v.passed for v in validations)

    def test_register_and_unregister_validator(self) -> None:
        """Test registering and unregistering a validator."""
        validator = ResultValidator()

        validator.register_validator("test", lambda r, e: (True, ""))

        expected = ExpectedResult(tool_name="test", custom_validator="test")
        assert validator.validate({}, expected).passed is True

        validator.unregister_validator("test")

        assert validator.validate({}, expected).passed is False


class TestBuiltInValidators:
    """Test built-in validator functions."""

    def test_validate_not_empty_with_none(self) -> None:
        """Test not_empty fails for None."""
        passed, msg = validate_not_empty(None, ExpectedResult(tool_name="test"))
        assert passed is False
        assert "None" in msg

    def test_validate_not_empty_with_empty_string(self) -> None:
        """Test not_empty fails for empty string."""
        passed, msg = validate_not_empty("   ", ExpectedResult(tool_name="test"))
        assert passed is False
        assert "empty" in msg

    def test_validate_not_empty_with_empty_list(self) -> None:
        """Test not_empty fails for empty list."""
        passed, msg = validate_not_empty([], ExpectedResult(tool_name="test"))
        assert passed is False

    def test_validate_not_empty_with_value(self) -> None:
        """Test not_empty passes for non-empty value."""
        passed, _ = validate_not_empty("data", ExpectedResult(tool_name="test"))
        assert passed is True

    def test_validate_is_dict_pass(self) -> None:
        """Test is_dict passes for dict."""
        passed, _ = validate_is_dict({"key": "value"}, ExpectedResult(tool_name="test"))
        assert passed is True

    def test_validate_is_dict_fail(self) -> None:
        """Test is_dict fails for non-dict."""
        passed, msg = validate_is_dict([1, 2, 3], ExpectedResult(tool_name="test"))
        assert passed is False
        assert "dict" in msg

    def test_validate_is_list_pass(self) -> None:
        """Test is_list passes for list."""
        passed, _ = validate_is_list([1, 2, 3], ExpectedResult(tool_name="test"))
        assert passed is True

    def test_validate_is_list_fail(self) -> None:
        """Test is_list fails for non-list."""
        passed, msg = validate_is_list({"key": "value"}, ExpectedResult(tool_name="test"))
        assert passed is False
        assert "list" in msg

    def test_validate_no_error_pass(self) -> None:
        """Test no_error passes when no error field."""
        passed, _ = validate_no_error({"status": "ok"}, ExpectedResult(tool_name="test"))
        assert passed is True

    def test_validate_no_error_fail(self) -> None:
        """Test no_error fails when error field exists."""
        passed, msg = validate_no_error(
            {"error": "Something went wrong"}, ExpectedResult(tool_name="test")
        )
        assert passed is False
        assert "error" in msg

    def test_validate_success_field_pass(self) -> None:
        """Test success_field passes when success is true."""
        passed, _ = validate_success_field(
            {"success": True}, ExpectedResult(tool_name="test")
        )
        assert passed is True

    def test_validate_success_field_fail_missing(self) -> None:
        """Test success_field fails when success field missing."""
        passed, msg = validate_success_field(
            {"data": "test"}, ExpectedResult(tool_name="test")
        )
        assert passed is False
        assert "missing" in msg

    def test_validate_success_field_fail_falsy(self) -> None:
        """Test success_field fails when success is false."""
        passed, msg = validate_success_field(
            {"success": False}, ExpectedResult(tool_name="test")
        )
        assert passed is False
        assert "falsy" in msg


class TestCreateDefaultValidator:
    """Test create_default_validator function."""

    def test_has_builtin_validators(self) -> None:
        """Test that default validator has built-ins registered."""
        validator = create_default_validator()

        # Test that built-in validators are available
        expected = ExpectedResult(tool_name="test", custom_validator="not_empty")
        result = validator.validate({"data": "test"}, expected)
        assert result.passed is True

        expected = ExpectedResult(tool_name="test", custom_validator="is_dict")
        result = validator.validate({"data": "test"}, expected)
        assert result.passed is True
