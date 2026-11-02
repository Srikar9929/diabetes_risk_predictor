"""Check that stacked lifestyle attributions roughly match the combined model drop."""

from __future__ import annotations

from planner import PlanBuilder

SAMPLE = {
    "pregnancies": 2,
    "glucose": 148.0,
    "blood_pressure": 82.0,
    "skin_thickness": 32.0,
    "insulin": 180.0,
    "bmi": 34.5,
    "diabetes_pedigree": 0.9,
    "age": 48,
    "sleep_hours": 5.5,
    "resting_hr": 84.0,
    "daily_steps": 4200,
    "active_minutes": 15,
    "hrv_ms": 32.0,
}


def main() -> int:
    plan = PlanBuilder().build(SAMPLE, top_k=4)
    item_sum = sum(item.estimated_risk_drop_pp for item in plan.items)
    total = plan.total_drop_pp
    print(f"baseline={plan.baseline_risk:.3f} projected={plan.projected_risk:.3f}")
    print(f"sum(item drops)={item_sum:.1f} pp  total_drop={total:.1f} pp")
    if abs(item_sum - total) > 0.15:
        print("FAIL: sequential item drops do not match combined plan total")
        return 1
    print("PASS: plan attribution is sequential and reconciles with total drop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
