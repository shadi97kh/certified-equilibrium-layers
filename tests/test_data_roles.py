"""Tests for the dataset role registry (real-data roles only)."""

from __future__ import annotations

import pytest

from cel.data import (
    EFFICACY_BACKBONE,
    OFFTARGET_CERTIFICATION,
    dataset_role,
    datasets_with_role,
)
from cel.data.roles import DATASET_ROLES, ROLES


def test_only_two_real_data_roles() -> None:
    # Per the hard data policy the data layer carries only real measured data, so
    # there are exactly two roles and no generated-data role.
    assert {EFFICACY_BACKBONE, OFFTARGET_CERTIFICATION} == ROLES
    assert all(role in ROLES for role in DATASET_ROLES.values())


def test_role_partitions_cover_the_registry() -> None:
    union = set(datasets_with_role(EFFICACY_BACKBONE)) | set(
        datasets_with_role(OFFTARGET_CERTIFICATION)
    )
    assert union == set(DATASET_ROLES)


def test_known_role_assignments() -> None:
    assert dataset_role("huesken") == EFFICACY_BACKBONE
    assert dataset_role("gse184929") == OFFTARGET_CERTIFICATION
    assert dataset_role("jackson2003") == OFFTARGET_CERTIFICATION
    assert dataset_role("seedseq") == OFFTARGET_CERTIFICATION


def test_unknown_role_and_dataset_raise() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        datasets_with_role("not_a_role")
    with pytest.raises(KeyError):
        dataset_role("does_not_exist")
