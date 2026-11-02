"""CPU-only batch CLI regressions, including real subprocesses and XLSX round trips."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from scripts import batch_semantic_contact_eval as batch


def read_workbook(path):
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(str(path)) as archive:
        # Every XML part must be well formed, including package relationships.
        for name in archive.namelist():
            ET.fromstring(archive.read(name))
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall("s:sheetData/s:row", namespace):
        values = [None] * len(batch.COLUMNS)
        for cell in row:
            column = ord(cell.get("r")[0]) - ord("A")
            if cell.get("t") == "inlineStr":
                values[column] = cell.find("s:is/s:t", namespace).text
            else:
                values[column] = float(cell.find("s:v", namespace).text)
        rows.append(values)
    return rows


class BatchEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = self.root / "ids.txt"
        self.ids.write_text("first\nsecond\n", encoding="utf-8")
        self.output = self.root / "results.xlsx"
        self.argv = ["--ids-file", str(self.ids), "--checkpoint", "policy.pth",
                     "--new-obj", "045", "--reference-dataset", "TACO", "--output", str(self.output)]
        self.task_values = [.5, 1.23456789, .75, 2.]

    def summary(self, **changes):
        result = {"status": "complete", "target_episodes": 120, "completed_episodes": 120,
                  "best_episode": dict(zip(batch.METRICS, (.123456789, .75, None)))}
        result.update(changes)
        path = self.root / "summary.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        return path

    def fake_project(self, task_metrics=True):
        package = self.root / "main" / "rl"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").touch()
        (package / "__init__.py").touch()
        (package / "train.py").write_text(textwrap.dedent('''\
            import json
            from pathlib import Path
            import sys

            options = dict(value.split("=", 1) for value in sys.argv[1:])
            data_id = json.loads(options["dataIndices"])[0]
            print("evaluating " + data_id, flush=True)
            if data_id == "failed":
                print("simulated failure", file=sys.stderr)
                sys.exit(7)
            experiment = json.loads(options["experiment"])
            directory = Path("dumps") / ("test_" + experiment + "__mock")
            directory.mkdir(parents=True)
            (directory / "command.json").write_text(json.dumps(sys.argv[1:]))
            count = int(options["test_episodes"])
            summary = {
                "status": "complete", "target_episodes": count, "completed_episodes": count,
                "best_episode": {
                    "operation_completion_ratio": .123456789,
                    "planned_semantic_contact_coverage": .75,
                    "black_violation_frame_ratio": None,
                },
                "semantic_contact": {
                    "operation_completion_ratio": .99,
                    "planned_semantic_contact_coverage": .99,
                    "black_violation_frame_ratio": .99,
                },
            }
            if WITH_TASK_METRICS:
                summary["task_metrics"] = {
                    "episodes": count, "successes": count // 2, "SR": .5,
                    "E_t": 1.23456789, "E_j": .75, "E_ft": 2., "error_unit": "cm",
                }
                if data_id == "different":
                    summary["task_metrics"].update(successes=0, SR=0., E_t=10.25, E_j=20.5, E_ft=30.75)
                print(("[test_task_summary] episodes: {episodes} SR: {SR:.4f} E_t: {E_t:.4f} "
                       "E_j: {E_j:.4f} E_ft: {E_ft:.4f} error_unit: {error_unit}").format(
                           **summary["task_metrics"]), flush=True)
            (directory / "summary.json").write_text(json.dumps(summary))
        ''').replace("WITH_TASK_METRICS", repr(task_metrics)), encoding="utf-8")
        for path in ("policy.pth", "assets/imitator_rh_artimano.pth", "assets/imitator_lh_artimano.pth"):
            checkpoint = self.root / path
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.touch()

    def test_ids_keep_file_order_bom_comments_and_duplicates(self):
        self.ids.write_text("\ufeff# header\n\nsecond # note\nfirst\nsecond\n", encoding="utf-8")
        self.assertEqual(batch.read_ids(self.ids), ["second", "first", "second"])
        for content in ("# empty\n", "first second\n", "../first\n"):
            self.ids.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                batch.read_ids(self.ids)

    def test_command_preserves_string_ids_object_and_paths_with_spaces(self):
        args = batch.parse_args(self.argv + ["--override", "+task.env.assetPathMap={/old:/new}"])
        args.checkpoint = "/some path/policy.pth"
        command = batch.build_command(args, "sequence@0", "unique_experiment")
        options = dict(value.split("=", 1) for value in command[5:])
        self.assertEqual(json.loads(options["dataIndices"]), ["sequence@0"])
        self.assertEqual(json.loads(options["task.env.newObj"]), "045")
        self.assertEqual(json.loads(options["checkpoint"]), args.checkpoint)
        self.assertEqual(options["headless"], "false")
        self.assertEqual(options["test_episodes"], "120")
        self.assertEqual(options["+task.env.assetPathMap"], "{/old:/new}")

    def test_aliases_and_object_normalization(self):
        argv = ["--newObj" if value == "--new-obj" else
                "--referenceDataset" if value == "--reference-dataset" else
                "45" if value == "045" else value for value in self.argv]
        self.assertEqual(batch.parse_args(argv).new_obj, "045")

    def test_invariants_cannot_be_disabled_with_extra_overrides(self):
        for override in ("test=false", "++test_episodes=0", "experiment=other", "~checkpoint",
                         "task.env.semanticContactEval.enabled=false", "hydra.run.dir=/tmp"):
            with self.subTest(override=override), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                batch.parse_args(self.argv + ["--override", override])

    def test_best_metrics_keep_precision_and_na(self):
        self.assertEqual(batch.load_best_metrics(self.summary(), 120), [.123456789, .75, "NA"])

    def test_task_metrics_keep_precision_and_validate_episode_count_and_units(self):
        metrics = dict(episodes=120, successes=60, SR=.5, E_t=1.23456789, E_j=.75, E_ft=2., error_unit="cm")
        self.assertEqual(batch.load_task_metrics(self.summary(task_metrics=metrics), 120), metrics)
        self.assertIsNone(batch.load_task_metrics(self.summary(), 120))
        self.assertIsNone(batch.load_task_metrics(self.summary(task_metrics=None), 120))
        for changes in ({"episodes": 119}, {"error_unit": "m"}, {"E_t": None}, {"E_j": -1.},
                        {"E_ft": float("nan")}, {"SR": 1.1}, {"SR": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                batch.load_task_metrics(self.summary(task_metrics=dict(metrics, **changes)), 120)

    def test_task_metrics_are_printed_and_exported_per_id_without_changing_semantic_report(self):
        self.fake_project()
        output = StringIO()
        with patch.object(batch, "ROOT", self.root), redirect_stdout(output):
            self.assertEqual(batch.main(self.argv), 0)
        expected = ("[test_task_summary] episodes: 120 SR: 0.5000 E_t: 1.2346 "
                    "E_j: 0.7500 E_ft: 2.0000 error_unit: cm")
        self.assertEqual(output.getvalue().splitlines().count(expected), 2)
        self.assertEqual(batch.COLUMNS, ("dataIndices", "new_obj", "operation_completion_ratio",
                                         "planned_semantic_contact_coverage", "black_violation_frame_ratio",
                                         "SR", "E_t", "E_j", "E_ft"))
        self.assertEqual(read_workbook(self.output), [list(batch.COLUMNS),
                         ["first", "045", .123456789, .75, "NA"] + self.task_values,
                         ["second", "045", .123456789, .75, "NA"] + self.task_values])
        report_path, = (self.root / "results_logs").glob("*/batch_report.json")
        for row in json.loads(report_path.read_text()):
            self.assertEqual(row["metrics"], dict(zip(batch.METRICS, (.123456789, .75, "NA"))))
            self.assertEqual(Path(row["process_log"]).read_text().splitlines().count(expected), 1)
            summary = json.loads(Path(row["summary"]).read_text())
            self.assertEqual(summary["task_metrics"]["E_t"], 1.23456789)

    def test_older_summaries_without_task_metrics_still_export_normally(self):
        self.fake_project(task_metrics=False)
        output = StringIO()
        with patch.object(batch, "ROOT", self.root), redirect_stdout(output):
            self.assertEqual(batch.main(self.argv), 0)
        self.assertNotIn("[test_task_summary]", output.getvalue())
        self.assertEqual(read_workbook(self.output)[1], ["first", "045", .123456789, .75, "NA"] + ["NA"] * 4)

    def test_each_id_exports_its_own_task_metrics_including_zero_success_rate(self):
        self.fake_project()
        self.ids.write_text("different\nfirst\n", encoding="utf-8")
        with patch.object(batch, "ROOT", self.root), redirect_stdout(StringIO()):
            self.assertEqual(batch.main(self.argv), 0)
        self.assertEqual(read_workbook(self.output)[1:], [
            ["different", "045", .123456789, .75, "NA", 0., 10.25, 20.5, 30.75],
            ["first", "045", .123456789, .75, "NA"] + self.task_values])

    def test_incomplete_or_missing_semantic_data_is_rejected(self):
        for change in ({"status": "incomplete"}, {"completed_episodes": 119}, {"target_episodes": 100},
                       {"best_episode": None}, {"best_episode": {}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                batch.load_best_metrics(self.summary(**change), 120)
        for value in (True, float("nan"), float("inf"), -.1, 1.1, "0.5"):
            best = dict(zip(batch.METRICS, (value, .75, .1)))
            with self.subTest(value=value), self.assertRaises(ValueError):
                batch.load_best_metrics(self.summary(best_episode=best), 120)
        path = self.root / "summary.json"
        for content in ("[]", "null", "not json"):
            path.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                batch.load_best_metrics(path, 120)

    def test_xlsx_keeps_text_ids_numbers_na_and_escapes_xml(self):
        rows = [["=sequence<&", "045", .123456789, "NA", 0.0] + self.task_values]
        batch.write_xlsx(self.output, rows)
        self.assertEqual(read_workbook(self.output), [list(batch.COLUMNS)] + rows)
        namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with ZipFile(str(self.output)) as archive:
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        self.assertEqual(sheet.find("s:autoFilter", namespace).get("ref"), "A1:I2")
        self.assertEqual(len(sheet.findall("s:cols/s:col", namespace)), 9)
        self.assertTrue(all("t" not in sheet.find("s:sheetData/s:row/s:c[@r='{}2']".format(column), namespace).attrib
                            for column in "FGHI"))
        batch.write_xlsx(self.output, [["updated", "045", 1, 0, 0, 0, 0, 0, 0]])
        self.assertEqual(read_workbook(self.output)[1], ["updated", "045", 1, 0, 0, 0, 0, 0, 0])

    def test_real_subprocesses_continue_after_failure_and_use_best_episode(self):
        self.fake_project()
        self.ids.write_text("second\nfailed\nfirst\n", encoding="utf-8")
        with patch.object(batch, "ROOT", self.root), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = batch.main(self.argv)
        self.assertEqual(code, 1)
        self.assertEqual(read_workbook(self.output), [list(batch.COLUMNS),
                         ["second", "045", .123456789, .75, "NA"] + self.task_values,
                         ["failed", "045"] + ["ERROR"] * 7,
                         ["first", "045", .123456789, .75, "NA"] + self.task_values])
        report_path, = (self.root / "results_logs").glob("*/batch_report.json")
        report = json.loads(report_path.read_text())
        self.assertEqual([row["dataIndices"] for row in report], ["second", "failed", "first"])
        self.assertEqual([row["status"] for row in report], ["complete", "error", "complete"])
        self.assertIn("code 7", report[1]["error"])
        self.assertIn("simulated failure", Path(report[1]["process_log"]).read_text())
        self.assertTrue(all("assetPathMap" not in arg for row in report for arg in row["command"]))

    def test_oakink_migration_reaches_real_subprocesses_and_exports_each_best_episode(self):
        self.fake_project()
        self.ids.write_text("9b728@0\n1b46d@1\n", encoding="utf-8")
        argv = list(self.argv)
        argv[argv.index("--new-obj") + 1] = "spoon@1"
        argv[argv.index("--reference-dataset") + 1] = "OakInk-v2"
        argv += ["--new-oakink", "data/migrated OakInk-v2",
                 "--override", "+task.env.assetPathMap={/other/assets:/copied/assets}"]
        with patch.object(batch, "ROOT", self.root), redirect_stdout(StringIO()):
            self.assertEqual(batch.main(argv), 0)
        self.assertEqual(read_workbook(self.output), [list(batch.COLUMNS),
                         ["9b728@0", "spoon@1", .123456789, .75, "NA"] + self.task_values,
                         ["1b46d@1", "spoon@1", .123456789, .75, "NA"] + self.task_values])
        commands = list((self.root / "dumps").glob("*/command.json"))
        self.assertEqual(len(commands), 2)
        expected_map = "+task.env.assetPathMap={" + batch.OLD_OAKINK + ":" + json.dumps(
            str(self.root / "data/migrated OakInk-v2")) + "}"
        for path in commands:
            command = json.loads(path.read_text())
            self.assertEqual(command[-1], expected_map)
            self.assertIn("+task.env.assetPathMap={/other/assets:/copied/assets}", command)

    def test_dry_run_accepts_absolute_new_oakink_root_alias_without_writing(self):
        destination = self.root / "copied assets" / "OakInk-v2"
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(batch.main(self.argv + ["--new-oakink-root", str(destination), "--dry-run"]), 0)
        self.assertIn(batch.OLD_OAKINK, output.getvalue())
        self.assertIn(str(destination), output.getvalue())
        self.assertFalse(self.output.exists())

    def test_shared_data_root_migration_runs_both_datasets_and_exports_best_episodes(self):
        self.fake_project()
        destination = self.root / "copied assets" / "data"
        for dataset, data_id, new_obj in (("TACO", "sk_box_1104218", "045"),
                                          ("OakInk-v2", "9b728@0", "spoon@1")):
            with self.subTest(dataset=dataset):
                self.ids.write_text(data_id + "\n", encoding="utf-8")
                argv = list(self.argv)
                argv[argv.index("--reference-dataset") + 1] = dataset
                argv[argv.index("--new-obj") + 1] = new_obj
                with patch.object(batch, "ROOT", self.root), redirect_stdout(StringIO()):
                    self.assertEqual(batch.main(argv + ["--new-data-root", str(destination)]), 0)
                self.assertEqual(read_workbook(self.output)[1],
                                 [data_id, new_obj, .123456789, .75, "NA"] + self.task_values)
        for path in (self.root / "dumps").glob("*/command.json"):
            command = json.loads(path.read_text())
            mapping = command[-1]
            for old_root in batch.DEFAULT_OLD_DATA_ROOTS:
                self.assertIn(old_root + ":" + json.dumps(str(destination)), mapping)

    def test_custom_old_data_roots_replace_defaults_and_reach_subprocess(self):
        self.fake_project()
        old_roots = ["/old server/data", "/old,archive/data"]
        argv = self.argv + ["--new-data", "data"]
        for old_root in old_roots:
            argv += ["--old-data-root", old_root]
        with patch.object(batch, "ROOT", self.root), redirect_stdout(StringIO()):
            self.assertEqual(batch.main(argv), 0)
        for path in (self.root / "dumps").glob("*/command.json"):
            mapping = json.loads(path.read_text())[-1]
            self.assertIn("/old\\ server/data", mapping)
            self.assertIn("/old\\,archive/data", mapping)
            self.assertIn(json.dumps(str(self.root / "data")), mapping)
            for old_root in batch.DEFAULT_OLD_DATA_ROOTS:
                self.assertNotIn(old_root, mapping)

    def test_ambiguous_or_relative_source_mapping_options_are_rejected(self):
        for options in (["--old-data-root", "/old/data"],
                        ["--old-data-root", "relative/data", "--new-data-root", "/new/data"],
                        ["--new-data-root", "/new/data", "--new-oakink", "/new/data/OakInk-v2"]):
            with self.subTest(options=options), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                batch.parse_args(self.argv + options)

    def test_successful_repeat_runs_have_isolated_logs(self):
        self.fake_project()
        with patch.object(batch, "ROOT", self.root), redirect_stdout(StringIO()):
            self.assertEqual(batch.main(self.argv), 0)
            self.assertEqual(batch.main(self.argv), 0)
        self.assertEqual(len(list((self.root / "results_logs").glob("*/batch_report.json"))), 2)
        self.assertEqual(len(list((self.root / "dumps").glob("*/summary.json"))), 4)

    def test_dry_run_needs_no_checkpoint_and_writes_nothing(self):
        with patch.object(batch, "run_evaluation") as run, redirect_stdout(StringIO()) as output:
            self.assertEqual(batch.main(self.argv + ["--dry-run"]), 0)
        run.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "results_logs").exists())
        self.assertLess(output.getvalue().index("[1/2] first"), output.getvalue().index("[2/2] second"))

    def test_interrupt_saves_completed_state_without_running_remaining_ids(self):
        self.fake_project()
        with patch.object(batch, "ROOT", self.root), patch.object(batch, "run_evaluation", side_effect=KeyboardInterrupt) as run:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(batch.main(self.argv), 130)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(read_workbook(self.output)[1:], [
            ["first", "045"] + ["ERROR"] * 7, ["second", "045"] + [None] * 7])


if __name__ == "__main__":
    unittest.main()
