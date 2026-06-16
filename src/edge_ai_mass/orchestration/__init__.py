"""Orchestration services for Jetson deployment."""

from edge_ai_mass.orchestration.config import InstallRule, OrchestratorConfig

__all__ = ["InstallRule", "Orchestrator", "OrchestratorConfig", "Updater"]


def __getattr__(name: str):
    if name == "Orchestrator":
        from edge_ai_mass.orchestration.start import Orchestrator

        return Orchestrator
    if name == "Updater":
        from edge_ai_mass.orchestration.updater import Updater

        return Updater
    raise AttributeError(name)
