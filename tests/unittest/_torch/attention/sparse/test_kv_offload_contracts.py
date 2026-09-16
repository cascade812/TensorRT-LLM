# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable CPU contract examples; these doubles do not validate CUDA providers.

Run this file directly with Python, or use pytest -c /dev/null --noconftest -p no:cacheprovider to
avoid the repository's GPU/native conftest. Source loading deliberately proves
the contracts do not depend on importing TensorRT-LLM, torch, or draft bindings.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.sparse import kv_offload_contracts as contracts
else:
    _path = (
        Path(__file__).resolve().parents[5]
        / "tensorrt_llm/_torch/attention/backends/sparse/kv_offload_contracts.py"
    )
    _spec = importlib.util.spec_from_file_location("_trtllm_kv_offload_contracts", _path)
    assert _spec is not None and _spec.loader is not None
    contracts = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = contracts
    _spec.loader.exec_module(contracts)


@dataclass
class _Completion:
    complete: bool = False
    waiters: list[contracts.StreamHandle] = field(default_factory=list)

    def wait_on(self, stream: contracts.StreamHandle) -> None:
        self.waiters.append(stream)

    def is_complete(self) -> bool:
        return self.complete


@dataclass(eq=False)
class _Reservation:
    planner: _Planner
    requirements: contracts.MemoryRequirements
    released: bool = False
    after: contracts.Completion | None = None

    def release(self, *, after: contracts.Completion | None = None) -> None:
        if not self.released:
            self.released = True
            self.after = after
            self.planner.collect()


class _Planner:
    """Small reference ledger for consumer tests, not a production scheduler.

    It knows allocation identities and bytes only. It deliberately has no
    knowledge of HiSparse slots, layouts, kernel limits, or allocation formulas.
    """

    def __init__(self, limits: dict[tuple[str, int], int]) -> None:
        self.limits = limits
        self.claims: list[_Reservation] = []

    def _totals(
        self, requirements: list[contracts.MemoryRequirements]
    ) -> dict[tuple[str, int], int]:
        allocations: dict[contracts.AllocationId, contracts.MemoryRequirement] = {}
        for requirements_set in requirements:
            for requirement in requirements_set.allocations:
                previous = allocations.get(requirement.allocation_id)
                if previous is not None and (
                    previous.memory,
                    previous.device_index,
                    previous.size_bytes,
                ) != (requirement.memory, requirement.device_index, requirement.size_bytes):
                    raise ValueError("Conflicting descriptions of one allocation")
                allocations[requirement.allocation_id] = requirement
        totals: dict[tuple[str, int], int] = {}
        for allocation in allocations.values():
            domain = (allocation.memory, allocation.device_index)
            totals[domain] = totals.get(domain, 0) + allocation.size_bytes
        for budget_domain, size in totals.items():
            if size > self.limits.get(budget_domain, 0):
                raise MemoryError(
                    f"{budget_domain}: need {size}, available {self.limits.get(budget_domain, 0)}"
                )
        return totals

    def reserve(self, requirements: contracts.MemoryRequirements) -> _Reservation:
        self._totals([c.requirements for c in self.claims] + [requirements])
        claim = _Reservation(self, requirements)
        self.claims.append(claim)
        return claim

    def replace(
        self, reservation: contracts.Reservation, requirements: contracts.MemoryRequirements
    ) -> None:
        if not isinstance(reservation, _Reservation) or reservation not in self.claims:
            raise ValueError("Unknown reservation")
        if reservation.released:
            raise RuntimeError("Cannot replace a retired reservation")
        self._totals([c.requirements for c in self.claims if c is not reservation] + [requirements])
        reservation.requirements = requirements

    def collect(self) -> None:
        self.claims = [
            c
            for c in self.claims
            if not c.released or (c.after is not None and not c.after.is_complete())
        ]

    def charged(self) -> dict[tuple[str, int], int]:
        return self._totals([c.requirements for c in self.claims])


@dataclass
class _HostLease:
    page: contracts.HostPageRef
    version: int
    valid_entries: int
    spans: tuple[contracts.HostSpan, ...]
    ready: _Completion
    retired: bool = False
    after: contracts.Completion | None = None

    def release(self, *, after: contracts.Completion) -> None:
        if not self.retired:
            self.retired = True
            self.after = after

    @property
    def pinned(self) -> bool:
        return not self.retired or self.after is None or not self.after.is_complete()


def _requirement(
    key: int, size: int, *, memory: str = "gpu", device: int = 0, namespace: str = "cache-0"
) -> contracts.MemoryRequirement:
    if memory == "host":
        return contracts.MemoryRequirement(
            contracts.AllocationId(namespace, key), "host", -1, size, "shared history"
        )
    return contracts.MemoryRequirement(
        contracts.AllocationId(namespace, key), "gpu", device, size, "cache and workspace"
    )


class TestCapacityContract(unittest.TestCase):
    def test_shared_host_storage_and_request_local_gpu_copies(self) -> None:
        planner = _Planner({("host", -1): 128, ("gpu", 0): 64})
        shared = _requirement(0, 128, memory="host", namespace="kvcm-0")
        a = planner.reserve(contracts.MemoryRequirements((shared, _requirement(1, 32))))
        b = planner.reserve(contracts.MemoryRequirements((shared, _requirement(2, 32))))
        self.assertEqual(planner.charged(), {("host", -1): 128, ("gpu", 0): 64})
        a.release()
        self.assertEqual(planner.charged(), {("host", -1): 128, ("gpu", 0): 32})
        b.release()
        self.assertEqual(planner.charged(), {})

    def test_exact_budget_and_rejection_are_atomic(self) -> None:
        planner: contracts.ReservationPlanner = _Planner({("gpu", 0): 64})
        claim = planner.reserve(contracts.MemoryRequirements((_requirement(0, 64),)))
        with self.assertRaises(MemoryError):
            planner.reserve(contracts.MemoryRequirements((_requirement(1, 1),)))
        self.assertEqual(claim.requirements.allocations[0].size_bytes, 64)
        claim.release()
        planner.reserve(contracts.MemoryRequirements((_requirement(2, 64),)))

    def test_failed_growth_preserves_old_claim(self) -> None:
        planner = _Planner({("gpu", 0): 64})
        old = contracts.MemoryRequirements((_requirement(0, 32),))
        claim = planner.reserve(old)
        with self.assertRaises(MemoryError):
            planner.replace(
                claim, contracts.MemoryRequirements((_requirement(0, 32), _requirement(1, 64)))
            )
        self.assertEqual(claim.requirements, old)
        self.assertEqual(planner.charged(), {("gpu", 0): 32})
        planner.replace(
            claim, contracts.MemoryRequirements((_requirement(0, 32), _requirement(1, 32)))
        )
        self.assertEqual(planner.charged(), {("gpu", 0): 64})

    def test_pending_release_cannot_be_reused(self) -> None:
        planner = _Planner({("gpu", 0): 64})
        claim = planner.reserve(contracts.MemoryRequirements((_requirement(0, 64),)))
        completion = _Completion()
        claim.release(after=completion)
        claim.release()  # A repeated call cannot discard the original dependency.
        planner.collect()
        with self.assertRaises(MemoryError):
            planner.reserve(contracts.MemoryRequirements((_requirement(1, 64),)))
        completion.complete = True
        planner.collect()
        planner.reserve(contracts.MemoryRequirements((_requirement(1, 64),)))

    def test_partial_submission_failure_keeps_its_reservation(self) -> None:
        planner = _Planner({("gpu", 0): 64})
        claim = planner.reserve(contracts.MemoryRequirements((_requirement(0, 64),)))
        completion = _Completion()
        error = contracts.SubmissionError("copy submission failed", completion)
        claim.release(after=error.completion)
        with self.assertRaises(MemoryError):
            planner.reserve(contracts.MemoryRequirements((_requirement(1, 64),)))
        completion.complete = True
        planner.collect()
        self.assertEqual(planner.charged(), {})

    def test_shared_storage_waits_for_every_claim(self) -> None:
        planner = _Planner({("host", -1): 64})
        requirements = contracts.MemoryRequirements((_requirement(0, 64, memory="host"),))
        a, b = planner.reserve(requirements), planner.reserve(requirements)
        completion = _Completion()
        a.release(after=completion)
        b.release()
        self.assertEqual(planner.charged(), {("host", -1): 64})
        completion.complete = True
        planner.collect()
        self.assertEqual(planner.charged(), {})

    def test_identity_conflicts_fail_before_accounting_changes(self) -> None:
        planner = _Planner({("gpu", 0): 128, ("gpu", 1): 128})
        planner.reserve(contracts.MemoryRequirements((_requirement(0, 32),)))
        for conflicting in (_requirement(0, 64), _requirement(0, 32, device=1)):
            with self.subTest(conflicting=conflicting), self.assertRaises(ValueError):
                planner.reserve(contracts.MemoryRequirements((conflicting,)))
        self.assertEqual(planner.charged(), {("gpu", 0): 32})

    def test_provider_namespaces_and_devices_are_independent(self) -> None:
        planner = _Planner({("gpu", 0): 32, ("gpu", 1): 32})
        planner.reserve(contracts.MemoryRequirements((_requirement(0, 32, namespace="a"),)))
        planner.reserve(
            contracts.MemoryRequirements((_requirement(0, 32, namespace="b", device=1),))
        )
        self.assertEqual(planner.charged(), {("gpu", 0): 32, ("gpu", 1): 32})


class TestHostLeaseContract(unittest.TestCase):
    def test_pending_backup_and_delayed_reader_have_distinct_completions(self) -> None:
        backup = _Completion()
        consumer = _Completion()
        lease = _HostLease(
            contracts.HostPageRef(17, contracts.LayerBinding(2, 7, 3), 0),
            1,
            4,
            (contracts.HostSpan(contracts.AllocationId("host", 0), 4096, 128),),
            backup,
        )
        reader: contracts.HostReadLease = lease
        reader.ready.wait_on(contracts.StreamHandle(3))
        self.assertFalse(reader.ready.is_complete())  # A stream wait never synchronizes the CPU.
        self.assertEqual(reader.valid_entries, 4)  # Coverage is conditional on readiness.
        backup.complete = True
        reader.release(after=consumer)
        reader.release(after=backup)  # Repeated release cannot shorten reader protection.
        self.assertTrue(lease.pinned)
        consumer.complete = True
        self.assertFalse(lease.pinned)


class TestDescriptors(unittest.TestCase):
    def test_compressed_entries_and_scales_use_page_byte_geometry(self) -> None:
        layout = contracts.EntryLayout(
            contracts.LayerBinding(2, 7, 3),
            entries_per_page=4,
            host_slot_bytes=(64, 16),
            components=(
                contracts.EntryComponent("kv", 0, 0, 16, 16),
                contracts.EntryComponent("scale", 1, 0, 4, 4),
            ),
        )
        self.assertEqual(layout.binding.layer_group_id, 3)
        self.assertEqual(sum(c.size for c in layout.components), 20)
        with self.assertRaises(ValueError):
            contracts.EntryLayout(layout.binding, 5, layout.host_slot_bytes, layout.components)

    def test_host_codec_coalesced_layer_offset_is_checked(self) -> None:
        binding = contracts.LayerBinding(0, 0, 5)
        component = contracts.EntryComponent("kv", 0, 32, 8, 8)
        contracts.EntryLayout(binding, 4, (64,), (component,))
        with self.assertRaises(ValueError):
            contracts.EntryLayout(binding, 4, (63,), (component,))

    def test_request_reserves_history_plus_generation(self) -> None:
        request = contracts.RequestSpec(17, history_tokens=128, reserved_tokens=160)
        self.assertEqual(request.reserved_tokens - request.history_tokens, 32)
        with self.assertRaises(ValueError):
            contracts.RequestSpec(17, history_tokens=128, reserved_tokens=127)

    def test_negative_bytes_and_invalid_budget_domains_fail(self) -> None:
        allocation = contracts.AllocationId("cache", 0)
        with self.assertRaises(ValueError):
            contracts.MemoryRequirement(allocation, "gpu", 0, -1, "invalid")
        with self.assertRaises(ValueError):
            contracts.MemoryRequirement(allocation, "host", 0, 1, "invalid")


if __name__ == "__main__":
    unittest.main()
