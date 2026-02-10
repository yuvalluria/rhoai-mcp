"""Domain registry for core RHOAI MCP plugins.

This module provides plugin classes for each core domain and a registry
function to instantiate them. All plugins use pluggy hooks for integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rhoai_mcp.hooks import hookimpl
from rhoai_mcp.plugin import BasePlugin, PluginMetadata

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.clients.base import CRDDefinition
    from rhoai_mcp.server import RHOAIServer


class ProjectsPlugin(BasePlugin):
    """Plugin for Data Science Project management."""

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="projects",
                version="0.1.0",
                description="Data Science Project management",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=[],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.projects.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_register_resources(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.projects.resources import register_resources

        register_resources(mcp, server)

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:  # noqa: ARG002
        return True, "Projects uses core Kubernetes and OpenShift APIs"


class NotebooksPlugin(BasePlugin):
    """Plugin for Workbench (Kubeflow Notebook) management."""

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="notebooks",
                version="0.1.0",
                description="Workbench (Kubeflow Notebook) management",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=["Notebook"],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.notebooks.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_get_crd_definitions(self) -> list[CRDDefinition]:
        from rhoai_mcp.domains.notebooks.crds import NotebookCRDs

        return [NotebookCRDs.NOTEBOOK]


class InferencePlugin(BasePlugin):
    """Plugin for Model Serving (KServe InferenceService) management."""

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="inference",
                version="0.1.0",
                description="Model Serving (KServe InferenceService) management",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=["InferenceService"],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.inference.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_get_crd_definitions(self) -> list[CRDDefinition]:
        from rhoai_mcp.domains.inference.crds import InferenceCRDs

        return [
            InferenceCRDs.INFERENCE_SERVICE,
            InferenceCRDs.SERVING_RUNTIME,
            InferenceCRDs.TEMPLATE,
        ]


class PipelinesPlugin(BasePlugin):
    """Plugin for Data Science Pipelines (DSPA) management."""

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="pipelines",
                version="0.1.0",
                description="Data Science Pipelines (DSPA) management",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=["DataSciencePipelinesApplication"],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.pipelines.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_get_crd_definitions(self) -> list[CRDDefinition]:
        from rhoai_mcp.domains.pipelines.crds import PipelinesCRDs

        return [PipelinesCRDs.DSPA]


class ConnectionsPlugin(BasePlugin):
    """Plugin for Data Connection (S3 secrets) management."""

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="connections",
                version="0.1.0",
                description="Data Connection (S3 secrets) management",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=[],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.connections.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:  # noqa: ARG002
        return True, "Data connections use core Kubernetes API"


class StoragePlugin(BasePlugin):
    """Plugin for Storage (PVC) management."""

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="storage",
                version="0.1.0",
                description="Storage (PVC) management",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=[],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.storage.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:  # noqa: ARG002
        return True, "Storage uses core Kubernetes API"


class TrainingPlugin(BasePlugin):
    """Plugin for Kubeflow Training Operator integration."""

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="training",
                version="0.1.0",
                description="Kubeflow Training Operator integration",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=["TrainJob", "ClusterTrainingRuntime"],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.training.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_get_crd_definitions(self) -> list[CRDDefinition]:
        from rhoai_mcp.domains.training.crds import TrainingCRDs

        return TrainingCRDs.all_crds()


class SummaryPlugin(BasePlugin):
    """Plugin for context-efficient cluster and project summaries.

    Provides lightweight tools optimized for AI agent context windows,
    offering compact overviews that reduce token usage significantly.
    """

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="summary",
                version="1.0.0",
                description="Context-efficient summary tools for AI agents",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=[],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.summary.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:  # noqa: ARG002
        return True, "Summary tools use core domain clients"


class PromptOptimizationPlugin(BasePlugin):
    """Plugin for prompt optimization using Opik.

    Provides MCP tools to run prompt evaluation and optimization via an
    external Opik-backed service (e.g. NeuralNav backend). Set
    RHOAI_MCP_OPIK_SERVICE_URL to the service base URL.
    """

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="prompt_optimization",
                version="0.1.0",
                description="Prompt evaluation and optimization using Opik",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=[],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.prompt_optimization.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:  # noqa: ARG002
        url = getattr(server.config, "opik_service_url", None)
        if url:
            return True, "Prompt optimization service URL configured"
        return True, "Prompt optimization tools available (set RHOAI_MCP_OPIK_SERVICE_URL to use)"


class NeuralNavPlugin(BasePlugin):
    """Plugin for NeuralNav deployment recommendation (model + GPU + agent + system prompt).

    Provides get_deployment_recommendation to fetch ranked recommendations from the
    NeuralNav backend. Uses the same RHOAI_MCP_OPIK_SERVICE_URL as prompt_optimization
    (NeuralNav backend exposes both /api/ranked-recommend-from-spec and evaluate/optimize).
    """

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="neuralnav",
                version="0.1.0",
                description="Deployment recommendation (model + GPU + agent + system prompt) via NeuralNav",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=[],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.domains.neuralnav.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:  # noqa: ARG002
        url = getattr(server.config, "opik_service_url", None)
        if url:
            return True, "NeuralNav service URL configured"
        return True, "NeuralNav tools available (set RHOAI_MCP_OPIK_SERVICE_URL to use)"


def get_core_plugins() -> list[BasePlugin]:
    """Return all core domain plugin instances.

    Returns:
        List of plugin instances for all core domains.
    """
    return [
        ProjectsPlugin(),
        NotebooksPlugin(),
        InferencePlugin(),
        PipelinesPlugin(),
        ConnectionsPlugin(),
        StoragePlugin(),
        TrainingPlugin(),
        SummaryPlugin(),
        PromptOptimizationPlugin(),
        NeuralNavPlugin(),
    ]
