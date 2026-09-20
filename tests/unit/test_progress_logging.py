from __future__ import annotations

import unittest

from progress_logging import progress_fields, progress_percent


class ProgressLoggingTests(unittest.TestCase):
    def test_percent_is_bounded_and_empty_workload_is_complete(self) -> None:
        self.assertEqual(progress_percent(1, 4), 25.0)
        self.assertEqual(progress_percent(-1, 4), 0.0)
        self.assertEqual(progress_percent(5, 4), 100.0)
        self.assertEqual(progress_percent(0, 0), 100.0)

    def test_fields_include_count_and_percentage(self) -> None:
        self.assertEqual(
            progress_fields(1, 3),
            "progress=1/3 progress_percent=33.33%",
        )


if __name__ == "__main__":
    unittest.main()
