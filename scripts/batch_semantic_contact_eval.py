#!/usr/bin/env python3
"""Run SPIDER checkpoint tests and export semantic and aggregate task metrics to XLSX."""

'''
python scripts/batch_semantic_contact_eval.py \
  --ids-file ids.txt \
  --checkpoint runs/SPIDER_Taco_sk_box_1104218_045_artimano__10-23-10-13-30/nn/last_SPIDER_Taco_sk_box_1104218_045_artimano_ep_10000_rew__141.53_.pth \
  --new-obj 045 \
  --reference-dataset TACO \
  --output add_data_evaluation/results.xlsx
'''

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_DATA_ROOTS = (
    "/media/user/89785a2b-45b3-43b5-99ad-a5558e7ad6484/"
    "spider_data/PhysGraph/data",
    "/home/user/ylj/spider_repro/spider/data",
)
OLD_OAKINK = DEFAULT_OLD_DATA_ROOTS[0] + "/OakInk-v2"
METRICS = (
    "operation_completion_ratio",
    "planned_semantic_contact_coverage",
    "black_violation_frame_ratio",
)
TASK_METRICS = ("SR", "E_t", "E_j", "E_ft")
COLUMNS = ("dataIndices", "new_obj") + METRICS + TASK_METRICS
# These options identify the run, guarantee finite semantic evaluation, or locate its logs.
RESERVED_OVERRIDES = {
    "dataIndices", "checkpoint", "experiment", "test", "test_episodes", "save_rollouts",
    "task.env.referenceDataset", "task.env.newObj", "task.env.semanticContactEval.enabled",
    "hydra.run.dir", "hydra.job.chdir",
}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-file", type=Path, required=True,
                        help="UTF-8 text file, one data ID per line; blank lines and # comments are ignored")
    parser.add_argument("--checkpoint", required=True, help="Shared residual-policy checkpoint; relative to the project")
    parser.add_argument("--new-obj", "--newObj", dest="new_obj", required=True)
    parser.add_argument("--reference-dataset", "--referenceDataset", dest="reference_dataset",
                        choices=("TACO", "OakInk-v2"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="Destination .xlsx file")
    parser.add_argument("--logs-dir", type=Path, help="Log parent; defaults to <output stem>_logs beside the workbook")
    parser.add_argument("--test-episodes", type=positive_int, default=120)
    parser.add_argument("--num-envs", type=positive_int, default=4)
    parser.add_argument("--headless", choices=("true", "false"), default="false")
    parser.add_argument("--reference-root", default="data/retargeting_spider_native")
    mapping = parser.add_mutually_exclusive_group()
    mapping.add_argument("--new-data-root", "--new-data", dest="new_data_root", type=Path,
                         help="New data directory containing OakInk-v2/ and TACO/; "
                              "relative paths are based on the project")
    parser.add_argument("--old-data-root", dest="old_data_roots", type=Path, action="append",
                        help="Old absolute data root recorded in manifests; repeatable, "
                             "defaults to the two data roots used by the existing bundles")
    mapping.add_argument("--new-oakink", "--new-oakink-root", dest="new_oakink", type=Path,
                         help="Legacy option mapping only OakInk-v2; "
                              "use --new-data-root for a shared data directory")
    parser.add_argument("--rh-checkpoint", default="assets/imitator_rh_artimano.pth")
    parser.add_argument("--lh-checkpoint", default="assets/imitator_lh_artimano.pth")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Additional Hydra override, repeatable (e.g. semanticMeshPaths or assetPathMap)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without evaluating or writing files")
    args = parser.parse_args(argv)
    if args.old_data_roots and args.new_data_root is None:
        parser.error("--old-data-root requires --new-data-root")
    for old_root in args.old_data_roots or []:
        if not old_root.expanduser().is_absolute():
            parser.error("--old-data-root must be an absolute source path from the manifest")
    if args.output.suffix.lower() != ".xlsx":
        parser.error("--output must end in .xlsx")
    if not args.new_obj or args.new_obj in (".", "..") or any(c in args.new_obj for c in "/\\"):
        parser.error("--new-obj must be a nonempty object ID without path separators")
    if args.reference_dataset == "TACO" and args.new_obj != "original":
        if not re.fullmatch(r"[0-9]{1,3}", args.new_obj):
            parser.error("TACO --new-obj must be a numeric ID from 000 to 999, or original")
        args.new_obj = args.new_obj.zfill(3)
    for override in args.override:
        key, separator, _ = override.partition("=")
        if not separator or not key or key.startswith("~"):
            parser.error("--override requires KEY=VALUE (deletions are unsupported)")
        if key.lstrip("+") in RESERVED_OVERRIDES:
            parser.error("{} is controlled by the batch runner; use its named options".format(key))
    return args


def read_ids(path):
    ids = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        data_id = line.partition("#")[0].strip()
        if not data_id:
            continue
        if data_id in (".", "..") or any(c.isspace() or c in "/\\" for c in data_id):
            raise ValueError("{}:{}: expected one data ID per line".format(path, line_number))
        ids.append(data_id)
    if not ids:
        raise ValueError("No data IDs found in {}".format(path))
    return ids


def hydra_string(value):
    # Quotes are part of the argument passed directly to Hydra, preserving e.g. "045".
    return json.dumps(str(value), ensure_ascii=False)


def hydra_dict_key(value):
    # Hydra dictionary keys use escaped literals rather than quoted strings.
    return re.sub(r"([\\(){}\[\]:=, \t])", r"\\\1", str(value))


def build_command(args, data_id, experiment):
    command = [
        sys.executable, "-u", "-m", "main.rl.train",
        "task=ResDexHand", "side=BiH", "bimanual_mode=united", "dexhand=artimano",
        "test=true", "headless=" + args.headless, "num_envs=" + str(args.num_envs),
        "randomStateInit=false", "rolloutStateInit=false",
        "dataIndices=[{}]".format(hydra_string(data_id)),
        "task.env.referenceSource=spider", "task.env.datasetType=spider",
        "task.env.referenceRoot=" + hydra_string(args.reference_root),
        "task.env.referenceDataset=" + hydra_string(args.reference_dataset),
        "task.env.newObj=" + hydra_string(args.new_obj),
        "task.env.objectReference=demonstration", "task.env.allowFailedReference=true",
        "task.env.episodeLength=1500", "task.task.randomize=false",
        "rh_base_model_checkpoint=" + hydra_string(args.rh_checkpoint),
        "lh_base_model_checkpoint=" + hydra_string(args.lh_checkpoint),
        "checkpoint=" + hydra_string(args.checkpoint),
        "experiment=" + hydra_string(experiment), "task.env.semanticContactEval.enabled=true",
        "test_episodes=" + str(args.test_episodes), "save_rollouts=false",
    ] + args.override
    mappings = {}
    if args.new_data_root is not None:
        new_root = (ROOT / args.new_data_root.expanduser()).resolve()
        for old_root in args.old_data_roots or DEFAULT_OLD_DATA_ROOTS:
            # Keep source paths lexical: resolving them on another server can
            # follow unrelated symlinks and no longer match the manifest.
            mappings[str(Path(old_root).expanduser())] = new_root
    elif args.new_oakink is not None:
        mappings[OLD_OAKINK] = (ROOT / args.new_oakink.expanduser()).resolve()
    if mappings:
        # Hydra merges dictionaries, retaining other --override asset mappings.
        entries = [hydra_dict_key(old) + ":" + hydra_string(new) for old, new in mappings.items()]
        command.append("+task.env.assetPathMap={" + ",".join(entries) + "}")
    return command


def load_complete_summary(summary_path, test_episodes):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("Expected a JSON object in {}".format(summary_path))
    if (summary.get("status") != "complete" or summary.get("target_episodes") != test_episodes
            or summary.get("completed_episodes") != test_episodes):
        raise ValueError("Incomplete or unexpected episode count in {}".format(summary_path))
    return summary


def load_best_metrics(summary_path, test_episodes):
    summary = load_complete_summary(summary_path, test_episodes)
    best = summary.get("best_episode")
    if not isinstance(best, dict):
        raise ValueError("Missing best_episode in {}".format(summary_path))
    values = []
    for metric in METRICS:
        if metric not in best:
            raise ValueError("Missing {} in best_episode: {}".format(metric, summary_path))
        value = best[metric]
        if value is None:
            values.append("NA")
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Invalid {}={!r} in {}".format(metric, value, summary_path))
        else:
            values.append(value)
    return values


def load_task_metrics(summary_path, test_episodes):
    summary = load_complete_summary(summary_path, test_episodes)
    task = summary.get("task_metrics")
    # Older summaries and tasks without diff_metrics retain the existing batch behavior.
    if task is None:
        return None
    if (not isinstance(task, dict) or task.get("episodes") != test_episodes
            or task.get("error_unit") != "cm"):
        raise ValueError("Invalid task_metrics in {}".format(summary_path))
    for name in TASK_METRICS:
        value = task.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (name == "SR" and value > 1)):
            raise ValueError("Invalid {}={!r} in task_metrics: {}".format(name, value, summary_path))
    return task


def write_xlsx(path, rows):
    """Write the nine-column workbook with stdlib OOXML; no extra Excel dependency."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    worksheet = ET.Element("worksheet", xmlns=ns)
    columns = ET.SubElement(worksheet, "cols")
    for index, width in enumerate((32, 24, 32, 42, 34) + (16,) * len(TASK_METRICS), 1):
        ET.SubElement(columns, "col", min=str(index), max=str(index), width=str(width), customWidth="1")
    sheet_data = ET.SubElement(worksheet, "sheetData")
    for row_index, values in enumerate([COLUMNS] + rows, 1):
        row = ET.SubElement(sheet_data, "row", r=str(row_index))
        for column, value in zip("ABCDEFGHI", values):
            if value is None:
                continue
            cell = ET.SubElement(row, "c", r="{}{}".format(column, row_index))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                ET.SubElement(cell, "v").text = repr(value)
            else:
                cell.set("t", "inlineStr")
                ET.SubElement(ET.SubElement(cell, "is"), "t").text = str(value)
    ET.SubElement(worksheet, "autoFilter", ref="A1:I{}".format(len(rows) + 1))
    parts = {
        "[Content_Types].xml": (
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '</Types>'
        ),
        "_rels/.rels": (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>'
        ),
        "xl/workbook.xml": (
            '<workbook xmlns="' + ns + '" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="best_episodes" sheetId="1" r:id="rId1"/></sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/></Relationships>'
        ),
        "xl/worksheets/sheet1.xml": ET.tostring(worksheet, encoding="utf-8", xml_declaration=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".xlsx", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        with ZipFile(str(temp_path), "w", compression=ZIP_DEFLATED) as archive:
            for name, content in parts.items():
                archive.writestr(name, content)
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def run_evaluation(command, process_log):
    # A fresh process per ID releases all simulator/GPU state between tests.
    with process_log.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Evaluation exited with code {}; see {}".format(result.returncode, process_log))


def main(argv=None):
    args = parse_args(argv)
    ids = read_ids(args.ids_file)
    if not args.dry_run:
        for checkpoint in (args.checkpoint, args.rh_checkpoint, args.lh_checkpoint):
            if not (ROOT / checkpoint).is_file():
                raise ValueError("Checkpoint does not exist: {}".format(ROOT / checkpoint))
    token = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    output = args.output.resolve()
    log_root = (args.logs_dir or output.with_name(output.stem + "_logs")).resolve() / token
    rows = [[data_id, args.new_obj] + [None] * (len(COLUMNS) - 2) for data_id in ids]
    reports = []
    if not args.dry_run:
        log_root.mkdir(parents=True)
        write_xlsx(output, rows)
    for index, data_id in enumerate(ids):
        label = re.sub(r"[^A-Za-z0-9_-]", "_", data_id)[:80]
        experiment = "batch_semantic_{}_{}_{:04d}_{}".format(args.reference_dataset, token, index + 1, label)
        command = build_command(args, data_id, experiment)
        print("[{}/{}] {} new_obj={}".format(index + 1, len(ids), data_id, args.new_obj), flush=True)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        run_logs = log_root / "{:04d}_{}".format(index + 1, label)
        run_logs.mkdir()
        report = {"dataIndices": data_id, "new_obj": args.new_obj, "command": command,
                  "process_log": str(run_logs / "process.log")}
        print("Process log: {}".format(report["process_log"]), flush=True)
        interrupted = False
        try:
            run_evaluation(command, Path(report["process_log"]))
            # The unique experiment prevents collecting a previous or concurrent run's summary.
            matches = list((ROOT / "dumps").glob("test_{}__*/summary.json".format(experiment)))
            if len(matches) != 1:
                raise ValueError("Expected one summary.json for {}, found {}".format(experiment, len(matches)))
            report["summary"] = str(matches[0])
            values = load_best_metrics(matches[0], args.test_episodes)
            task_metrics = load_task_metrics(matches[0], args.test_episodes)
            task_values = ([task_metrics[name] for name in TASK_METRICS]
                           if task_metrics is not None else ["NA"] * len(TASK_METRICS))
            rows[index][2:] = values + task_values
            report.update(status="complete", metrics=dict(zip(METRICS, values)))
            print("Best episode: {}".format(dict(zip(METRICS, values))), flush=True)
            if task_metrics is not None:
                print(("[test_task_summary] episodes: {episodes} SR: {SR:.4f} E_t: {E_t:.4f} "
                       "E_j: {E_j:.4f} E_ft: {E_ft:.4f} error_unit: {error_unit}").format(**task_metrics),
                      flush=True)
        except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            rows[index][2:] = ["ERROR"] * (len(COLUMNS) - 2)
            report.update(status="interrupted" if interrupted else "error", error=str(exc) or "KeyboardInterrupt")
            print("ERROR {}: {}".format(data_id, report["error"]), file=sys.stderr, flush=True)
        reports.append(report)
        write_xlsx(output, rows)
        (log_root / "batch_report.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if interrupted:
            return 130
    if args.dry_run:
        return 0
    failures = sum(report["status"] != "complete" for report in reports)
    print("Saved {} ({} succeeded, {} failed). Logs: {}".format(output, len(ids) - failures, failures, log_root), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
