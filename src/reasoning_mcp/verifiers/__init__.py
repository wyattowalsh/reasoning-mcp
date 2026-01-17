"""Reasoning verifiers (Process Reward Models) for step validation.

Verifiers score and validate reasoning steps, providing process rewards
to guide search and improve answer quality.
"""

from reasoning_mcp.verifiers.base import VerifierBase, VerifierMetadata
from reasoning_mcp.verifiers.gar_discriminator import GAR_DISCRIMINATOR_METADATA, GarDiscriminator
from reasoning_mcp.verifiers.gen_prm import GEN_PRM_METADATA, GenPrm
from reasoning_mcp.verifiers.or_prm import OR_PRM_METADATA, OrPrm
from reasoning_mcp.verifiers.r_prm import R_PRM_METADATA, RPrm
from reasoning_mcp.verifiers.rrm import RRM_METADATA, Rrm
from reasoning_mcp.verifiers.think_prm import THINK_PRM_METADATA, ThinkPrm
from reasoning_mcp.verifiers.versa_prm import VERSA_PRM_METADATA, VersaPrm

# All verifier classes
VERIFIERS = {
    "think_prm": ThinkPrm,
    "gen_prm": GenPrm,
    "r_prm": RPrm,
    "versa_prm": VersaPrm,
    "rrm": Rrm,
    "gar_discriminator": GarDiscriminator,
    "or_prm": OrPrm,
}

# All verifier metadata
VERIFIER_METADATA = {
    "think_prm": THINK_PRM_METADATA,
    "gen_prm": GEN_PRM_METADATA,
    "r_prm": R_PRM_METADATA,
    "versa_prm": VERSA_PRM_METADATA,
    "rrm": RRM_METADATA,
    "gar_discriminator": GAR_DISCRIMINATOR_METADATA,
    "or_prm": OR_PRM_METADATA,
}

__all__ = [
    # Base
    "VerifierBase",
    "VerifierMetadata",
    # Verifiers
    "ThinkPrm",
    "GenPrm",
    "RPrm",
    "VersaPrm",
    "Rrm",
    "GarDiscriminator",
    "OrPrm",
    # Metadata
    "THINK_PRM_METADATA",
    "GEN_PRM_METADATA",
    "R_PRM_METADATA",
    "VERSA_PRM_METADATA",
    "RRM_METADATA",
    "GAR_DISCRIMINATOR_METADATA",
    "OR_PRM_METADATA",
    # Registries
    "VERIFIERS",
    "VERIFIER_METADATA",
]
