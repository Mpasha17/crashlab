# Note: sql-analyst fault smokes on gpt-oss-20b (2026-10-23)

After `output_parse_failed` recovery. One question (March revenue). Observational.

| fault | outcome | note |
|---|---|---|
| clean | **3/3 pass** | recovery returns prose with correct number |
| spec_drop | **fail** | recovered prose had **848.99** (wrong); oracle correctly fails |
| tool_5xx | **fail** | apology / no correct answer |
| schema_drift | **pass** | recovered 198.99 |
| misleading_desc | **fail** | confused cents prose |

Takeaway: prose recovery does **not** invent oracle passes — wrong recovered
numbers still fail `answer_matches_ground_truth`.
