"""Reasoning ensemblers for multi-model combination and aggregation.

Ensemblers combine multiple models or reasoning paths to improve
accuracy and robustness through diverse aggregation strategies.
"""

from reasoning_mcp.ensemblers.base import EnsemblerBase, EnsemblerMetadata
from reasoning_mcp.ensemblers.der import DER_METADATA, Der
from reasoning_mcp.ensemblers.ema_fusion import EMA_FUSION_METADATA, EmaFusion
from reasoning_mcp.ensemblers.moa import MOA_METADATA, Moa
from reasoning_mcp.ensemblers.model_switch import MODEL_SWITCH_METADATA, ModelSwitch
from reasoning_mcp.ensemblers.multi_agent_verification import (
    MULTI_AGENT_VERIFICATION_METADATA,
    MultiAgentVerification,
)
from reasoning_mcp.ensemblers.slm_mux import SLM_MUX_METADATA, SlmMux
from reasoning_mcp.ensemblers.training_free_orchestration import (
    TRAINING_FREE_ORCHESTRATION_METADATA,
    TrainingFreeOrchestration,
)

# All ensembler classes
ENSEMBLERS = {
    "der": Der,
    "moa": Moa,
    "slm_mux": SlmMux,
    "multi_agent_verification": MultiAgentVerification,
    "ema_fusion": EmaFusion,
    "model_switch": ModelSwitch,
    "training_free_orchestration": TrainingFreeOrchestration,
}

# All ensembler metadata
ENSEMBLER_METADATA = {
    "der": DER_METADATA,
    "moa": MOA_METADATA,
    "slm_mux": SLM_MUX_METADATA,
    "multi_agent_verification": MULTI_AGENT_VERIFICATION_METADATA,
    "ema_fusion": EMA_FUSION_METADATA,
    "model_switch": MODEL_SWITCH_METADATA,
    "training_free_orchestration": TRAINING_FREE_ORCHESTRATION_METADATA,
}

__all__ = [
    # Base
    "EnsemblerBase",
    "EnsemblerMetadata",
    # Ensemblers
    "Der",
    "Moa",
    "SlmMux",
    "MultiAgentVerification",
    "EmaFusion",
    "ModelSwitch",
    "TrainingFreeOrchestration",
    # Metadata
    "DER_METADATA",
    "MOA_METADATA",
    "SLM_MUX_METADATA",
    "MULTI_AGENT_VERIFICATION_METADATA",
    "EMA_FUSION_METADATA",
    "MODEL_SWITCH_METADATA",
    "TRAINING_FREE_ORCHESTRATION_METADATA",
    # Registries
    "ENSEMBLERS",
    "ENSEMBLER_METADATA",
]
