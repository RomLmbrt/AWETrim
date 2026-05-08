"""Aerostructural interfaces and adapters."""

from awetrim.aerostructural.pss.coupling import PssKineticDampingSolver, PssQsmCoupler
from awetrim.aerostructural.protocols import (
    AeroToStructuralLoadMapper,
    AeroToStructureMap,
    AerodynamicGeometryUpdate,
    DeformableAeroBody,
    PssStructuralSolver,
    PssSystem,
    QsmAerostructuralCoupler,
    QsmCouplingRequest,
    QsmCouplingResult,
    QsmCouplingSettings,
    QsmIterationRecord,
    StructuralGeometry,
    StructuralToAeroMapper,
    TapeActuationState,
)

__all__ = [
    "AeroToStructuralLoadMapper",
    "AeroToStructureMap",
    "AerodynamicGeometryUpdate",
    "DeformableAeroBody",
    "PssKineticDampingSolver",
    "PssQsmCoupler",
    "PssStructuralSolver",
    "PssSystem",
    "QsmAerostructuralCoupler",
    "QsmCouplingRequest",
    "QsmCouplingResult",
    "QsmCouplingSettings",
    "QsmIterationRecord",
    "StructuralGeometry",
    "StructuralToAeroMapper",
    "TapeActuationState",
]
