"""Tests for Model Registry auth utilities."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.config import ModelRegistryAuthMode
from rhoai_mcp.domains.model_registry.auth import (
    _get_in_cluster_token,
    _get_oauth_token_from_kubeconfig,
    _is_running_in_cluster,
    build_auth_headers,
)


class TestIsRunningInCluster:
    """Test _is_running_in_cluster helper function."""

    def test_in_cluster_token_exists(self) -> None:
        """When service account token exists, we're in-cluster."""
        with patch("pathlib.Path.exists", return_value=True):
            assert _is_running_in_cluster() is True

    def test_outside_cluster_token_missing(self) -> None:
        """When service account token doesn't exist, we're outside cluster."""
        with patch("pathlib.Path.exists", return_value=False):
            assert _is_running_in_cluster() is False


class TestGetInClusterToken:
    """Test _get_in_cluster_token helper function."""

    def test_token_file_exists(self, tmp_path: Any) -> None:
        """When token file exists, return its contents."""
        token_file = tmp_path / "token"
        token_file.write_text("my-service-account-token")

        with patch("rhoai_mcp.domains.model_registry.auth.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = "my-service-account-token\n"
            token = _get_in_cluster_token()

        assert token == "my-service-account-token"

    def test_token_file_missing(self) -> None:
        """When token file doesn't exist, return None."""
        with patch("rhoai_mcp.domains.model_registry.auth.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            token = _get_in_cluster_token()

        assert token is None


class TestGetOAuthTokenFromKubeconfig:
    """Test _get_oauth_token_from_kubeconfig helper function."""

    def test_kubeconfig_not_found(self) -> None:
        """When kubeconfig doesn't exist, return None."""
        mock_config = MagicMock()
        mock_config.effective_kubeconfig_path.exists.return_value = False

        token = _get_oauth_token_from_kubeconfig(mock_config)

        assert token is None

    def test_kubeconfig_with_token(self) -> None:
        """When kubeconfig has a token, return it."""
        mock_config = MagicMock()
        mock_config.effective_kubeconfig_path.exists.return_value = True
        mock_config.kubeconfig_context = None

        with patch(
            "kubernetes.config.kube_config.KubeConfigLoader"
        ) as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.token = "sha256~my-oauth-token"
            mock_loader_class.return_value = mock_loader

            token = _get_oauth_token_from_kubeconfig(mock_config)

        assert token == "sha256~my-oauth-token"

    def test_kubeconfig_without_token(self) -> None:
        """When kubeconfig has no token, return None."""
        mock_config = MagicMock()
        mock_config.effective_kubeconfig_path.exists.return_value = True
        mock_config.kubeconfig_context = None

        with patch(
            "kubernetes.config.kube_config.KubeConfigLoader"
        ) as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.token = None
            mock_loader_class.return_value = mock_loader

            token = _get_oauth_token_from_kubeconfig(mock_config)

        assert token is None

    def test_kubeconfig_loading_error(self) -> None:
        """When kubeconfig loading fails, return None."""
        mock_config = MagicMock()
        mock_config.effective_kubeconfig_path.exists.return_value = True
        mock_config.kubeconfig_context = None

        with patch(
            "kubernetes.config.kube_config.KubeConfigLoader"
        ) as mock_loader_class:
            mock_loader_class.side_effect = Exception("Config error")
            token = _get_oauth_token_from_kubeconfig(mock_config)

        assert token is None


class TestBuildAuthHeaders:
    """Test build_auth_headers function."""

    @pytest.fixture
    def mock_config_none(self) -> MagicMock:
        """Create a mock config with no auth."""
        config = MagicMock()
        config.model_registry_auth_mode = ModelRegistryAuthMode.NONE
        config.model_registry_token = None
        return config

    @pytest.fixture
    def mock_config_token(self) -> MagicMock:
        """Create a mock config with token auth."""
        config = MagicMock()
        config.model_registry_auth_mode = ModelRegistryAuthMode.TOKEN
        config.model_registry_token = "my-explicit-token"
        return config

    @pytest.fixture
    def mock_config_oauth(self) -> MagicMock:
        """Create a mock config with OAuth auth."""
        config = MagicMock()
        config.model_registry_auth_mode = ModelRegistryAuthMode.OAUTH
        config.model_registry_token = None
        config.effective_kubeconfig_path.exists.return_value = True
        config.kubeconfig_context = None
        return config

    def test_no_auth_returns_empty_headers(self, mock_config_none: MagicMock) -> None:
        """When auth mode is NONE, no headers are returned."""
        headers = build_auth_headers(mock_config_none)
        assert headers == {}

    def test_token_auth_adds_bearer_header(self, mock_config_token: MagicMock) -> None:
        """When auth mode is TOKEN, Authorization header is added."""
        headers = build_auth_headers(mock_config_token)
        assert headers["Authorization"] == "Bearer my-explicit-token"

    def test_token_auth_no_token_configured(self) -> None:
        """When auth mode is TOKEN but no token configured, empty headers."""
        config = MagicMock()
        config.model_registry_auth_mode = ModelRegistryAuthMode.TOKEN
        config.model_registry_token = None

        headers = build_auth_headers(config)
        assert headers == {}

    def test_oauth_auth_outside_cluster(self, mock_config_oauth: MagicMock) -> None:
        """When auth mode is OAUTH outside cluster, gets token from kubeconfig."""
        with patch(
            "rhoai_mcp.domains.model_registry.auth._is_running_in_cluster",
            return_value=False,
        ), patch(
            "rhoai_mcp.domains.model_registry.auth._get_oauth_token_from_kubeconfig",
            return_value="sha256~kubeconfig-token",
        ):
            headers = build_auth_headers(mock_config_oauth)

        assert headers["Authorization"] == "Bearer sha256~kubeconfig-token"

    def test_oauth_auth_inside_cluster(self, mock_config_oauth: MagicMock) -> None:
        """When auth mode is OAUTH inside cluster, gets token from SA."""
        with patch(
            "rhoai_mcp.domains.model_registry.auth._is_running_in_cluster",
            return_value=True,
        ), patch(
            "rhoai_mcp.domains.model_registry.auth._get_in_cluster_token",
            return_value="sa-token-123",
        ):
            headers = build_auth_headers(mock_config_oauth)

        assert headers["Authorization"] == "Bearer sa-token-123"

    def test_oauth_auth_no_token_found(self, mock_config_oauth: MagicMock) -> None:
        """When auth mode is OAUTH but no token found, returns empty headers."""
        with patch(
            "rhoai_mcp.domains.model_registry.auth._is_running_in_cluster",
            return_value=False,
        ), patch(
            "rhoai_mcp.domains.model_registry.auth._get_oauth_token_from_kubeconfig",
            return_value=None,
        ):
            headers = build_auth_headers(mock_config_oauth)

        assert headers == {}

    def test_requires_auth_override_triggers_oauth(
        self, mock_config_none: MagicMock
    ) -> None:
        """When requires_auth_override is True, OAuth is attempted even if mode is NONE."""
        with patch(
            "rhoai_mcp.domains.model_registry.auth._is_running_in_cluster",
            return_value=False,
        ), patch(
            "rhoai_mcp.domains.model_registry.auth._get_oauth_token_from_kubeconfig",
            return_value="override-token",
        ):
            headers = build_auth_headers(mock_config_none, requires_auth_override=True)

        assert headers["Authorization"] == "Bearer override-token"

    def test_requires_auth_override_without_token(
        self, mock_config_none: MagicMock
    ) -> None:
        """When requires_auth_override is True but no token, returns empty headers."""
        with patch(
            "rhoai_mcp.domains.model_registry.auth._is_running_in_cluster",
            return_value=False,
        ), patch(
            "rhoai_mcp.domains.model_registry.auth._get_oauth_token_from_kubeconfig",
            return_value=None,
        ):
            headers = build_auth_headers(mock_config_none, requires_auth_override=True)

        assert headers == {}
