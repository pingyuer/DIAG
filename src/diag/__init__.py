"""DIAG: Dynamics-Induced Affine Gating for echo video segmentation."""
from diag.anchoring import (
    BACKBONE_RECORD,
    CONTENT_KEY_COARSE,
    CONTENT_KEY_FINE,
    ContentAnchor,
)
from diag.decoder import CandidateDecoder
from diag.ds_head import DsHead
from diag.hdc import HDC, adjacent_ratio_cv, compose_gain
from diag.losses import DiagLoss, DiagLossWeights
from diag.pclf import PCLF
from diag.svf import DiffeomorphicTransform, SVFWarpHead

__all__ = [
    "BACKBONE_RECORD",
    "CONTENT_KEY_COARSE",
    "CONTENT_KEY_FINE",
    "HDC",
    "PCLF",
    "CandidateDecoder",
    "ContentAnchor",
    "DiagLoss",
    "DiagLossWeights",
    "DiffeomorphicTransform",
    "DsHead",
    "SVFWarpHead",
    "adjacent_ratio_cv",
    "compose_gain",
]


def main() -> None:
    print("diag anchor,pclf,hdc,decoder,ds,svf")
