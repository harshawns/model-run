"""Check the public example's claims against its emitted trace."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

class ResumeDemoTest(unittest.TestCase):
    def test_trace_distinguishes_answers_from_execution_failure(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('resume_demo.py'))], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        trace = json.loads(result.stdout)
        self.assertEqual(trace['cases']['correct']['total_correct'], 2)
        self.assertEqual(trace['cases']['wrong']['total_correct'], 0)
        self.assertEqual(trace['cases']['malformed']['total_correct'], 0)
        self.assertTrue(trace['cases']['wrong']['done'])
        self.assertEqual(trace['cases']['verifier_error']['error_type'], 'RuntimeError')
        self.assertIsNone(trace['cases']['verifier_error']['reward'])
        self.assertFalse(trace['cases']['verifier_error']['recorded_as_scored_answer'])
        for obs in trace['observations']:
            self.assertNotIn('answer', obs)
            self.assertNotIn('gold_answer', obs)
        self.assertTrue(trace['fixture_gold_not_in_reset_observation'])

if __name__ == '__main__':
    unittest.main()
