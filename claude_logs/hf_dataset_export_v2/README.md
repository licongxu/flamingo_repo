---
license: cc-by-4.0
language:
  - en
tags:
  - agent-trace
  - claude
  - cosmology
  - reasoning-trace
  - llm-agent
pretty_name: Fable 5 FLAMINGO Research Task Traces
size_categories:
  - n<1K
configs:
  - config_name: task_index
    data_files:
      - split: train
        path: viewer/task_index.parquet
  - config_name: fable5_actions_by_task
    data_files:
      - split: train
        path: viewer/fable5_actions_by_task.parquet
  - config_name: task1_joint_cnc_masked_tsz_project_plan_trace
    data_files:
      - split: train
        path: viewer/task1_joint_cnc_masked_tsz_project_plan_trace.parquet
  - config_name: task1_repository_exploration_subagent_trace
    data_files:
      - split: train
        path: viewer/task1_repository_exploration_subagent_trace.parquet
  - config_name: task1_joint_cnc_masked_tsz_project_plan_timeline
    data_files:
      - split: train
        path: viewer/task1_joint_cnc_masked_tsz_project_plan_timeline.parquet
  - config_name: task2_l1m9_rotation_group_power_spectrum_trace
    data_files:
      - split: train
        path: viewer/task2_l1m9_rotation_group_power_spectrum_trace.parquet
  - config_name: task2_l1m9_rotation_group_power_spectrum_timeline
    data_files:
      - split: train
        path: viewer/task2_l1m9_rotation_group_power_spectrum_timeline.parquet
  - config_name: task3_l1m9_rotation_group_commit_push_trace
    data_files:
      - split: train
        path: viewer/task3_l1m9_rotation_group_commit_push_trace.parquet
  - config_name: task3_l1m9_rotation_group_commit_push_timeline
    data_files:
      - split: train
        path: viewer/task3_l1m9_rotation_group_commit_push_timeline.parquet
---

# Fable 5 FLAMINGO Research Task Traces

This dataset is a small corpus of **Claude Fable 5 / Claude Code traces** from
real FLAMINGO cosmology-analysis work. It is organized by explicit research task
so the Dataset Viewer makes clear what was being operated on and how the agent
took actions.

Start with the `task_index` config. It lists each task, the exact original user
prompt when available, the model setup, what Claude did, the raw trace file, and
the corresponding viewer configs.

Then use `fable5_actions_by_task` for a compact action stream. It keeps every
record keyed by task and labels records as `tool_use`, `tool_result`,
`assistant_reasoning_or_response`, `user_instruction_or_command_output`,
`file_snapshot`, or environment/session metadata. The full original record is
still preserved in `record_json`.

## Tasks

| Task | Viewer config | Raw trace | Description |
|---|---|---|---|
| `2026-07-06_fable5_joint_cnc_masked_tsz_plan` | `task1_joint_cnc_masked_tsz_project_plan_trace` | `fable5_session_trace.jsonl` | Repository survey and publication-oriented project plan for joint cluster number counts plus masked tSZ power-spectrum analysis on FLAMINGO. |
| `2026-07-06_fable5_joint_cnc_masked_tsz_plan` subagent | `task1_repository_exploration_subagent_trace` | `trace_subagents/explore_repo_subagent.jsonl` | Subagent sweep of notebooks and repository context for Task 1. |
| `2026-07-08_fable5_L1_m9_rotation_group_ps` | `task2_l1m9_rotation_group_power_spectrum_trace` | `fable5_L1_m9_rotation_group_trace.jsonl` | Implementation/run trace for creating the L1_m9 rotation-group tSZ power-spectrum analysis. |
| `2026-07-08_fable5_L1_m9_rotation_group_commit_push` | `task3_l1m9_rotation_group_commit_push_trace` | `fable5_L1_m9_rotation_group_commit_push_trace.jsonl` | Separate Fable 5 session for the same L1_m9 prompt, including the nb40 pipeline commit `fc69fa6` and push to GitHub. |

## Viewer Tables

All task trace tables use the same browsable schema:

- `idx`: record number within that source trace.
- `type`: original record kind, such as `user`, `assistant`, `attachment`, or `file-history-snapshot`.
- `timestamp`: original timestamp when present.
- `role`: message role when present.
- `text`: readable flattened content, including labelled tool calls/results.
- `record_json`: complete original JSON record as a string.

`fable5_actions_by_task` adds task/action columns:

- `task_id`, `task_title`, `source`.
- `action_type`, `tool_name`, `summary`, `paths_mentioned`.
- `record_json` for the lossless original record.

For Task 2, the exact L1_m9 prompt is in `task_index.original_user_prompt` and
also appears in the raw trace at `idx=14`. For Task 3, the repeated L1_m9
prompt appears at raw trace `idx=17`, followed by commit and push actions.

## Canonical Raw Artifacts

The raw `.jsonl` files are the source of truth. Parquet files in `viewer/` are
only convenience views for the Hugging Face Dataset Viewer.

## Provenance and Safety

- Model: Claude Fable 5, run through Claude Code.
- Domain: FLAMINGO cosmology simulation analysis.
- The traces were scanned for common API key/token patterns before upload. They
  may contain local HPC filesystem paths such as `/scratch/...` and `/rds/...`.
- Scientific claims in traces are model-generated research/workflow artifacts,
  not peer-reviewed measurements.
