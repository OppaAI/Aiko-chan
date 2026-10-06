"""Standalone stdlib regression tests for the Phase 1 evaluation harness."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

HARNESS = Path(__file__).resolve().parents[2] / "conscience-lab" / "harness"
sys.path.insert(0, str(HARNESS))
import adapter
import run_eval
import screen_model


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dev = self.root / "dev.jsonl"
        self.out = self.root / "results"
        self.gguf = self.root / "candidate.gguf"
        self.gguf.write_bytes(b"synthetic candidate")
        self.write_dev(2)

    def write_dev(self, n):
        rows = [{"id": f"case-{i}", "checkpoint": "input", "situation": "synthetic text",
                 "expect": "refuse" if i % 2 else "allow",
                 "refusal_category": "violence" if i % 2 else None} for i in range(n)]
        self.dev.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def evaluate(self, *args):
        argv = ["run_eval.py", "--dev", str(self.dev), "--backend", "dummy",
                "--out", str(self.out), *args]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            run_eval.main()
        return json.loads((self.out / "summary.json").read_text())

    def test_parse_failures_never_refuse(self):
        classifier = adapter.ZeroShotAdapter([
            {"name": "violence", "handling": "TERMINAL", "definition": "synthetic"}])
        for raw, category, ok in [(" violence\n", "violence", True), ("NONE", None, True),
                                   ("violence because...", None, False),
                                   ("not violence", None, False), ("", None, False),
                                   ("unknown", None, False)]:
            with self.subTest(raw=raw):
                result = classifier.classify(adapter.DummyBackend(lambda _: raw), "text", random.Random(0))
                self.assertEqual(result["predicted_category"], category)
                self.assertEqual(result["predicted_refuse"], category is not None)
                self.assertEqual(result["parse_ok"], ok)

    def test_invalid_prevalence_rejected_before_reading_or_writing(self):
        for value in ["nan", "inf", "-inf", "-0.01", "1.01"]:
            with self.subTest(value=value), patch.object(run_eval, "load_categories") as load:
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as error:
                        self.evaluate(f"--prevalence={value}")
                self.assertEqual(error.exception.code, 2)
                self.assertIn("--prevalence must be finite", stderr.getvalue())
                load.assert_not_called()
                self.assertFalse(self.out.exists())

    def test_valid_prevalence_including_endpoints(self):
        for value in ["0", "0.05", "1"]:
            with self.subTest(value=value):
                summary = self.evaluate(f"--prevalence={value}")
                self.assertEqual(summary["prevalence_assumed"], float(value))

    def test_backend_failure_aborts_each_loop_without_outputs(self):
        prediction = {"predicted_category": None, "predicted_refuse": False,
                      "parse_ok": True, "raw": "none"}
        for big_benign in [False, True]:
            with self.subTest(big_benign=big_benign):
                # Fail after successful predictions to catch partial-run publication.
                effects = [prediction] * (3 if big_benign else 1) + [RuntimeError("offline")]
                with patch.object(adapter.ZeroShotAdapter, "classify", side_effect=effects) as classify:
                    with self.assertRaisesRegex(SystemExit, "big-benign" if big_benign else "dev"):
                        self.evaluate(*(["--benign-file", str(self.dev)] if big_benign else []))
                self.assertEqual(classify.call_count, len(effects))
                self.assertFalse(self.out.exists())

    def test_nearest_rank_p95(self):
        for n, expected in [(1, 1), (2, 2), (10, 10), (20, 19), (21, 20)]:
            with self.subTest(n=n):
                self.write_dev(n)
                times = [0]
                for milliseconds in range(n, 0, -1):
                    times.extend([0, milliseconds / 1000])
                times.append(1)
                with patch.object(run_eval.time, "time", side_effect=times):
                    summary = self.evaluate()
                self.assertEqual(summary["model_metrics"]["latency_ms_p95"], expected)

    def screen(self, props, content="none", model=""):
        responses = [io.BytesIO(json.dumps(props).encode()),
                     io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())]
        argv = ["screen_model.py", "--gguf", str(self.gguf), "--server", "http://server:8080",
                "--model", model]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as output:
            with patch.object(screen_model.urllib.request, "urlopen", side_effect=responses) as request:
                code = screen_model.main()
        return code, output.getvalue(), request

    def test_matching_server_model_and_nonempty_answer_pass(self):
        code, output, request = self.screen({"model_path": str(self.gguf)}, " none\n", "candidate/alias")
        self.assertEqual(code, 0)
        self.assertIn("PASS: server inference", output)
        self.assertEqual(request.call_args_list[0].args[0], "http://server:8080/props?model=candidate%2Falias")
        payload = json.loads(request.call_args_list[1].args[0].data)
        self.assertEqual(payload["model"], "candidate/alias")

    def test_unverified_server_model_never_runs_inference(self):
        for path in [None, "", "candidate.gguf", "/other/candidate.gguf", "/models/other.gguf", 42]:
            with self.subTest(path=path):
                code, output, request = self.screen({"model_path": path}, model="candidate")
                self.assertEqual(code, 1)
                self.assertNotIn("PASS: server inference", output)
                self.assertEqual(request.call_count, 1)
        code, _, request = self.screen({"model_alias": "candidate"})
        self.assertEqual(code, 1)
        self.assertEqual(request.call_count, 1)

    def test_empty_or_nontext_answer_fails(self):
        for content in ["", " \n\t", None, 42]:
            with self.subTest(content=content):
                code, output, request = self.screen({"model_path": str(self.gguf)}, content)
                self.assertEqual(code, 1)
                self.assertNotIn("PASS: server inference", output)
                self.assertEqual(request.call_count, 2)

    def test_server_errors_fail_gate(self):
        argv = ["screen_model.py", "--gguf", str(self.gguf), "--server", "http://server:8080"]
        for effects in [[OSError("offline")],
                        [io.BytesIO(json.dumps({"model_path": str(self.gguf)}).encode()), OSError("offline")],
                        [io.BytesIO(json.dumps({"model_path": str(self.gguf)}).encode()), io.BytesIO(b"{}")]]:
            with self.subTest(effects=effects), patch.object(sys, "argv", argv):
                with patch.object(screen_model.urllib.request, "urlopen", side_effect=effects):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(screen_model.main(), 1)
                self.assertNotIn("PASS: server inference", output.getvalue())


if __name__ == "__main__":
    unittest.main()
