"""Authentication utilities for Model Registry clients.

This module provides shared authentication functions used by both the
ModelRegistryClient and ModelCatalogClient to avoid code duplication.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rhoai_mcp.config import RHOAIConfig

logger = logging.getLogger(__name__)


def _is_running_in_cluster() -> bool:
    """Check if we're running inside a Kubernetes cluster.

    Returns:
        True if running in-cluster (as a pod), False otherwise.
    """
    return Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists()


def _get_in_cluster_token() -> str | None:
    """Get the service account token when running in-cluster.

    Returns:
        The service account token, or None if not running in-cluster.
    """
    token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    if token_path.exists():
        try:
            return token_path.read_text().strip()
        except Exception as e:
            logger.debug(f"Error reading in-cluster token: {e}")
    return None


def _get_oauth_token_from_kubeconfig(config: RHOAIConfig) -> str | None:
    """Extract the OAuth token from kubeconfig.

    This retrieves the bearer token that was set when the user logged in
    via 'oc login' or 'kubectl login'. The token is used for authenticating
    with OpenShift services protected by OAuth proxy.

    Args:
        config: RHOAI configuration with kubeconfig settings.

    Returns:
        The OAuth bearer token, or None if not found.
    """
    try:
        from kubernetes import config as k8s_config  # type: ignore[import-untyped]

        # Determine which kubeconfig to use
        kubeconfig_path = config.effective_kubeconfig_path
        if not kubeconfig_path.exists():
            logger.debug(f"Kubeconfig not found: {kubeconfig_path}")
            return None

        # Load the kubeconfig
        loader = k8s_config.kube_config.KubeConfigLoader(
            config_file=str(kubeconfig_path),
            active_context=config.kubeconfig_context,
        )

        # Get the token from the current context's user credentials
        # The token is stored in the user's auth-provider or directly
        token: str | None = loader.token
        if token:
            logger.debug("Retrieved OAuth token from kubeconfig")
            return str(token)

        logger.debug("No token found in kubeconfig")
        return None

    except Exception as e:
        logger.debug(f"Error extracting OAuth token from kubeconfig: {e}")
        return None


def build_auth_headers(
    config: RHOAIConfig,
    requires_auth_override: bool = False,
) -> dict[str, str]:
    """Build authentication headers for Model Registry API calls.

    This function consolidates the auth header construction logic used by
    ModelRegistryClient, ModelCatalogClient, and the probe_api_type function.

    Args:
        config: RHOAI configuration with auth settings.
        requires_auth_override: If True, attempt OAuth auth even if auth_mode is NONE.
            Used when discovery indicates the endpoint requires authentication.

    Returns:
        Dict of headers to include in requests.
    """
    from rhoai_mcp.config import ModelRegistryAuthMode

    auth_mode = config.model_registry_auth_mode
    headers: dict[str, str] = {}

    # Check if auth is required
    if auth_mode == ModelRegistryAuthMode.NONE and not requires_auth_override:
        return headers

    token: str | None = None

    if auth_mode == ModelRegistryAuthMode.TOKEN:
        # Use explicit token from config
        token = config.model_registry_token
        if not token:
            logger.warning(
                "Model Registry auth_mode is 'token' but no token configured. "
                "Set RHOAI_MCP_MODEL_REGISTRY_TOKEN environment variable."
            )

    elif auth_mode == ModelRegistryAuthMode.OAUTH or requires_auth_override:
        # Get OAuth token from kubeconfig or in-cluster SA
        if _is_running_in_cluster():
            token = _get_in_cluster_token()
        else:
            token = _get_oauth_token_from_kubeconfig(config)

        if not token:
            logger.warning(
                "Model Registry auth_mode is 'oauth' but no token found. "
                "Ensure you are logged in (oc login) or running in-cluster."
            )

    if token:
        headers["Authorization"] = f"Bearer {token}"
        logger.debug("Added Authorization header for Model Registry")

    return headers
