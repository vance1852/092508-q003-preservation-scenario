from __future__ import annotations

import json
import sqlite3
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from collection_logistics.api import JsonApplication
from collection_logistics.clock import FrozenClock
from collection_logistics.errors import Conflict, Forbidden, ScenarioInputMissing
from collection_logistics.planning import AllocationRequest, RiskPoint, allocate_capacity, latest_streak
from collection_logistics.service import CollectionLogisticsService
from collection_logistics.risk import DemandBucket, inventory_coverage, mark_to_risk, traffic_gap


class PlanningTests(unittest.TestCase):
    def test_latest_down_streak_uses_first_close_as_base(self) -> None:
        streak = latest_streak([
            RiskPoint("2026-09-18", Decimal("108")),
            RiskPoint("2026-09-19", Decimal("105")),
            RiskPoint("2026-09-20", Decimal("102")),
            RiskPoint("2026-09-21", Decimal("98")),
        ])
        self.assertEqual(streak.direction, "down")
        self.assertEqual(streak.sessions, 4)
        self.assertEqual(streak.start_date, "2026-09-18")
        self.assertEqual(streak.end_close, Decimal("98"))

    def test_allocation_is_stable_and_does_not_exceed_capacity(self) -> None:
        rows = allocate_capacity(Decimal("100"), [
            AllocationRequest("later", Decimal("80"), 20, "2026-09-24T09:00:00Z"),
            AllocationRequest("first", Decimal("70"), 10, "2026-09-24T10:00:00Z"),
        ])
        self.assertEqual(rows[0]["dispatch_id"], "first")
        self.assertEqual(rows[0]["allocated_units"], "70.000")
        self.assertEqual(rows[1]["allocated_units"], "30.000")

    def test_inventory_coverage_and_traffic_gap(self) -> None:
        coverage = inventory_coverage(
            [{"center_id": "receiving-vault", "preservation_resource_kind": "tow-truck", "available_units": "250"}],
            [DemandBucket("receiving-vault", "tow-truck", Decimal("100"), Decimal("20"))],
        )
        self.assertEqual(coverage[0]["coverage_days"], "2.30")
        self.assertTrue(coverage[0]["below_three_days"])
        gap = traffic_gap(
            opening_inventory=Decimal("100"),
            confirmed_inbound=Decimal("30"),
            forecast_demand=Decimal("120"),
            protected_reserve=Decimal("40"),
        )
        self.assertEqual(gap["traffic_gap"], "30.000")

    def test_mark_to_risk_groups_deterministically(self) -> None:
        result = mark_to_risk(
            [{"position_id": "p1", "risk_index": "HUMIDITY", "quantity_units": "100", "baseline_value": "105"}],
            {"HUMIDITY": Decimal("98")},
        )
        self.assertEqual(result["unrealized_pnl_cny"], "-700.00")


class CollectionLogisticsServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = CollectionLogisticsService(self.connection, self.clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            self.service.create_user(user_id, user_id, role)
        self.service.create_facility("plan", {"center_id": "collection-east", "name": "北部标本事件保藏中心", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_units": "500000"})
        self.service.create_facility("plan", {"center_id": "receiving-vault-b", "name": "沿海终端", "kind": "receiving-vault", "timezone": "Asia/Shanghai", "capacity_units": "800000"})
        self.service.create_route("plan", {"corridor_id": "transfer-east-1", "origin_center_id": "collection-east", "destination_center_id": "receiving-vault-b", "preservation_resource_kind": "preservation-box", "hourly_capacity": "100000", "delay_basis_points": 25, "response_minutes": 36})

    def tearDown(self) -> None:
        self.connection.close()

    def risk_record(self, day: int, close: str) -> dict[str, object]:
        return self.service.record_risk_record("plan", {"risk_index": "HUMIDITY", "duty_date": f"2026-09-{day}", "index_value": close, "source_revision": f"r-{day}", "observed_at": f"2026-09-{day}T21:00:00Z"})

    def test_risk_record_revisions_preserve_history(self) -> None:
        first = self.risk_record(23, "98")
        second = self.service.record_risk_record("plan", {"risk_index": "HUMIDITY", "duty_date": "2026-09-23", "index_value": "97.8", "source_revision": "r-23-corrected", "observed_at": "2026-09-23T22:00:00Z"})
        self.assertNotEqual(first["risk_record_id"], second["risk_record_id"])
        rows = self.connection.execute("SELECT * FROM risk_index_risk_records ORDER BY risk_record_id").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["supersedes_risk_record_id"], rows[0]["risk_record_id"])

    def test_dispatch_request_replay_and_payload_conflict(self) -> None:
        payload = {"dispatch_id": "nom-1", "corridor_id": "transfer-east-1", "specimen_event_id": "herbarium-room", "duty_date": "2026-09-25", "requested_units": "80000", "priority": 10, "idempotency_key": "key-1"}
        first = self.service.submit_dispatch("dispatch", payload)
        self.assertEqual(first, self.service.submit_dispatch("dispatch", payload))
        changed = dict(payload, requested_units="81000")
        with self.assertRaises(Conflict):
            self.service.submit_dispatch("dispatch", changed)

    def test_outage_reduces_allocation_and_deployment_consumes_inventory(self) -> None:
        self.service.announce_restriction("risk", "transfer-east-1", "2026-09-25T00:00:00Z", "2026-09-25T23:59:59Z", "50", "检修")
        for number, requested, priority in ((1, "40000", 10), (2, "30000", 20)):
            self.service.submit_dispatch("dispatch", {"dispatch_id": f"nom-{number}", "corridor_id": "transfer-east-1", "specimen_event_id": f"specimen_event-{number}", "duty_date": "2026-09-25", "requested_units": requested, "priority": priority, "idempotency_key": f"key-{number}"})
        allocation = self.service.allocate("dispatch", "transfer-east-1", "2026-09-25")
        self.assertEqual(allocation["available_units"], "50000.000")
        self.assertEqual(allocation["allocations"][1]["allocated_units"], "10000.000")
        self.service.add_inventory_lot("dispatch", {"preservation_resource_lot_id": "lot-1", "center_id": "collection-east", "preservation_resource_kind": "preservation-box", "grade": "HUMIDITY", "quantity_units": "60000", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})
        deployment = self.service.dispatch_deployment("dispatch", "deployment-1", "nom-1", "lot-1", 2)
        self.assertEqual(deployment["deployed_units"], "40000.000")
        self.assertEqual(self.service.inventory_lot("lot-1")["available_units"], "20000.000")

    def test_scenario_is_approved_and_replayed_by_input(self) -> None:
        self.risk_record(23, "98")
        self.service.add_inventory_lot("dispatch", {"preservation_resource_lot_id": "lot-1", "center_id": "collection-east", "preservation_resource_kind": "preservation-box", "grade": "HUMIDITY", "quantity_units": "60000", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})
        self.service.create_scenario("plan", {"scenario_id": "restart", "name": "库房环境恢复", "risk_index": "HUMIDITY", "risk_index_drop_percent": "9", "route_capacity_changes": {"transfer-east-1": "20"}, "demand_changes": {"collection-east:preservation-box": "-5"}})
        with self.assertRaises(Forbidden):
            self.service.approve_scenario("plan", "restart", 1)
        self.service.approve_scenario("risk", "restart", 1)
        first = self.service.run_scenario("plan", "restart", "2026-09-23")
        second = self.service.run_scenario("plan", "restart", "2026-09-23")
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["run_id"], second["run_id"])

    def _approved_humidity_scenario(self, scenario_id: str = "restart", drop: str = "9") -> None:
        self.service.create_scenario("plan", {"scenario_id": scenario_id, "name": "库房环境恢复", "risk_index": "HUMIDITY", "risk_index_drop_percent": drop, "route_capacity_changes": {}, "demand_changes": {}})
        self.service.approve_scenario("risk", scenario_id, 1)

    def test_scenario_picks_bound_series_not_latest_record_of_day(self) -> None:
        self.risk_record(23, "98")
        # 同日更晚录入虫害压力指数：旧逻辑会把 37 当作湿度基准。
        self.service.record_risk_record("plan", {"risk_index": "INJURY", "duty_date": "2026-09-23", "index_value": "37", "source_revision": "pest-23", "observed_at": "2026-09-23T23:00:00Z"})
        self._approved_humidity_scenario("dry-plan")
        result = self.service.run_scenario("plan", "dry-plan", "2026-09-23")
        # 98 * (1 - 9/100) = 89.18，而不是 37 * 0.91。
        self.assertEqual(result["projected_risk_index_cny"], "89.18")
        used = result["risk_index_input"]
        self.assertEqual(used["risk_index"], "HUMIDITY")
        self.assertEqual(used["duty_date"], "2026-09-23")
        self.assertEqual(used["index_value"], "98")
        self.assertEqual(used["source_revision"], "r-23")
        self.assertEqual(used["observed_at"], "2026-09-23T21:00:00Z")
        self.assertEqual(used["selected_as_of"], "2026-09-23")

    def test_scenario_uses_revision_effective_at_as_of_date(self) -> None:
        self.risk_record(23, "98")
        self._approved_humidity_scenario("dry-plan")
        first = self.service.run_scenario("plan", "dry-plan", "2026-09-23")
        self.assertEqual(first["risk_index_input"]["index_value"], "98")
        # 事后更正湿度（新修订版本），历史方案重放不得漂移。
        self.service.record_risk_record("plan", {"risk_index": "HUMIDITY", "duty_date": "2026-09-23", "index_value": "50", "source_revision": "r-23-fix", "observed_at": "2026-09-25T10:00:00Z"})
        replayed = self.service.run_scenario("plan", "dry-plan", "2026-09-23")
        self.assertTrue(replayed["replayed"])
        self.assertEqual(replayed["run_id"], first["run_id"])
        self.assertEqual(replayed["projected_risk_index_cny"], "89.18")
        self.assertEqual(replayed["risk_index_input"]["source_revision"], "r-23")
        self.assertEqual(replayed["risk_index_input"]["index_value"], "98")
        # 另一日期首次运行则选择当时有效的新修订版本。
        later = self.service.run_scenario("plan", "dry-plan", "2026-09-24")
        self.assertFalse(later["replayed"])
        self.assertEqual(later["risk_index_input"]["source_revision"], "r-23-fix")
        self.assertEqual(later["risk_index_input"]["index_value"], "50")

    def test_scenario_without_bound_series_returns_business_error(self) -> None:
        self.risk_record(23, "98")
        self.service.create_scenario("plan", {"scenario_id": "pest-plan", "name": "虫害处置", "risk_index": "INJURY", "risk_index_drop_percent": "10", "route_capacity_changes": {}, "demand_changes": {}})
        self.service.approve_scenario("risk", "pest-plan", 1)
        with self.assertRaises(ScenarioInputMissing) as ctx:
            self.service.run_scenario("plan", "pest-plan", "2026-09-23")
        self.assertEqual(ctx.exception.code, "scenario_input_missing")
        self.assertIn("INJURY", str(ctx.exception))
        # 补齐系列后可以运行。
        self.service.record_risk_record("plan", {"risk_index": "INJURY", "duty_date": "2026-09-22", "index_value": "12", "source_revision": "pest-22", "observed_at": "2026-09-22T20:00:00Z"})
        result = self.service.run_scenario("plan", "pest-plan", "2026-09-23")
        self.assertEqual(result["risk_index_input"]["risk_index"], "INJURY")
        self.assertEqual(result["risk_index_input"]["index_value"], "12")

    def test_scenario_requires_bound_risk_index(self) -> None:
        with self.assertRaises(Exception):
            self.service.create_scenario("plan", {"scenario_id": "no-index", "name": "缺指标", "risk_index_drop_percent": "9", "route_capacity_changes": {}, "demand_changes": {}})

    def test_legacy_run_without_frozen_input_still_replays_original_result(self) -> None:
        from collection_logistics.planning import canonical_json
        # 修复前的旧情景定义没有 risk_index 字段，旧运行结果也没有来源信息。
        self.connection.execute(
            "INSERT INTO response_scenarios(scenario_id,name,definition_json,content_sha256,state,created_by,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            ("legacy-scn", "旧情景", canonical_json({"scenario_id": "legacy-scn"}), "a" * 64, "approved", "risk", "2026-09-20T00:00:00Z"),
        )
        old_result = {"projected_risk_index_cny": "89.18", "road_corridors": [], "inventory": []}
        self.connection.execute(
            "INSERT INTO response_scenario_runs(scenario_id,as_of_date,input_sha256,result_json,created_by,created_at) "
            "VALUES(?,?,?,?,?,?)",
            ("legacy-scn", "2026-09-23", "b" * 64, canonical_json(old_result), "plan", "2026-09-23T22:00:00Z"),
        )
        # 即使之后补录了任意指标数据，旧方案仍按原结果重放。
        self.risk_record(23, "98")
        replayed = self.service.run_scenario("plan", "legacy-scn", "2026-09-23")
        self.assertTrue(replayed["replayed"])
        self.assertEqual(replayed["projected_risk_index_cny"], "89.18")

    def test_audit_chain_detects_tampering(self) -> None:
        self.assertTrue(self.service.audit_chain("audit")["valid"])
        self.connection.execute("UPDATE traffic_audit_events SET payload_json='{}' WHERE event_id=1")
        self.assertFalse(self.service.audit_chain("audit")["valid"])

    def test_api_exposes_browser_free_boundary(self) -> None:
        app = JsonApplication(self.service)
        self.assertEqual(app.handle("GET", "/health").status, 200)
        response = app.handle("GET", "/risk_records/summary/HUMIDITY", {"X-Actor-Id": "plan"})
        self.assertEqual(response.status, 404)
        self.assertEqual(response.body["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
