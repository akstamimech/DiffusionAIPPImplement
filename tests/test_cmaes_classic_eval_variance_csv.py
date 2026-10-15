import csv
import tempfile
import unittest
from pathlib import Path

from CMAES_classic_eval import write_variance_trace


class VarianceTraceCsvTests(unittest.TestCase):
    def test_write_variance_trace_writes_one_total_variance_per_timestep(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "variance_over_time.csv"

            write_variance_trace([10.0, 8.5, 7.25], output_path)

            with output_path.open(newline="") as trace_file:
                rows = list(csv.reader(trace_file))

        self.assertEqual(rows[0], ["total_variance"])
        self.assertEqual([float(row[0]) for row in rows[1:]], [10.0, 8.5, 7.25])


if __name__ == "__main__":
    unittest.main()
