"""The models every chaperone stage calls, in one place.

Kept here rather than as a constant in each module because the IDs went stale
once already: five modules each carried their own string, so updating "the
model" meant finding all five. Override either with an environment variable
to pin a run to a specific version.

AGENT_MODEL runs the per-candidate triage loop and the fold reconsideration —
long multi-tool conversations where the reasoning is the product.
AUDIT_MODEL runs the passes that check another model's output: the gate
reconsideration, the validation-strategy designer and the review agent.
"""
import os

AGENT_MODEL = os.environ.get("CHAPERONE_AGENT_MODEL", "claude-fable-5-2")
AUDIT_MODEL = os.environ.get("CHAPERONE_AUDIT_MODEL", "claude-sonnet-5-5")
