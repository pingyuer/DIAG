"""DIAG: Dynamics-Induced Affine Gating for echo video segmentation."""
from diag.anchoring import BACKBONE_RECORD, CONTENT_KEY_COARSE, CONTENT_KEY_FINE, ContentAnchor
from diag.decoder import CandidateDecoder
from diag.ds_head import DsHead
from diag.hdc import HDC, adjacent_ratio_cv, compose_gain
from diag.losses import DiagLoss, DiagLossWeights
from diag.pclf import PCLF
from diag.svf import DiffeomorphicTransform, SVFWarpHead

__all__ = [
    "ContentAnchor", "BACKBONE_RECORD", "CONTENT_KEY_FINE", "CONTENT_KEY_COARSE",
    "PCLF", "HDC", "compose_gain", "adjacent_ratio_cv",
    "CandidateDecoder", "DiagLoss", "DiagLossWeights",
    "DsHead", "DiffeomorphicTransform", "SVFWarpHead",
]


def main() -> None:
    print("diag " + ",".join(["anchor", "pclf", "hdc", "decoder", "ds", "svf"]))
