# 🧭 Human + Machine Gate Workflow Example

This document illustrates how the new **task leasing** model works in the
`pipeline_control` table, replacing the old `process_status` pattern.

---

## 📋 Table Schema Summary

| Column | Purpose | Controlled By |
|---------|----------|---------------|
| `status` | Machine lifecycle (`NEW → IN_PROGRESS → COMPLETED → ERROR`) | FSM / runners |
| `task_state` | Human gate (`READY`, `PAUSED`, `SKIP`) | Human/admin |
| `is_claimed` | Whether a runner currently owns the task | Lease system |
| `worker_id` | ID of the active worker | Lease system |
| `lease_until` | Timestamp when the claim expires | Lease system |
| `decision_flag` | FSM transition or branching signal | FSM manager |

---

## 🧩 Example Scenario

Two jobs are ready for processing:

| url | iteration | stage | status | task_state | is_claimed | worker_id | lease_until |
|---|---:|---|---|---|---|---|---|
| `A` | 1 | SIMILARITY_METRICS | NEW | READY | FALSE | NULL | NULL |
| `B` | 1 | SIMILARITY_METRICS | NEW | READY | FALSE | NULL | NULL |

Two workers start up:

- **Worker A** → `worker_id = "sim-a1b2c3d4"`
- **Worker B** → `worker_id = "sim-e5f6g7h8"`

Default lease time = 15 minutes.

---

## 1️⃣ Build Candidate List

Both workers run:

```sql
SELECT url, iteration
FROM pipeline_control
WHERE stage='SIMILARITY_METRICS'
  AND status='NEW'
  AND task_state='READY'
  AND (is_claimed=FALSE OR lease_until IS NULL OR lease_until < CURRENT_TIMESTAMP);
