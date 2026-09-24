from __future__ import annotations

import unittest

from birdly.inference.predict import Prediction, filter_predictions_by_confidence


def _predictions(*scores: float) -> list[Prediction]:
    return [
        {
            "species_name": f"Species {index}",
            "taxon_id": index,
            "score": score,
        }
        for index, score in enumerate(scores)
    ]


class PredictionFilterTests(unittest.TestCase):
    def test_top_confidence_below_threshold_returns_empty(self) -> None:
        self.assertEqual(
            filter_predictions_by_confidence(_predictions(0.48, 0.30)),
            [],
        )

    def test_top_confidence_at_threshold_returns_empty(self) -> None:
        self.assertEqual(
            filter_predictions_by_confidence(_predictions(0.50, 0.30)),
            [],
        )

    def test_one_prediction_reaches_target(self) -> None:
        result = filter_predictions_by_confidence(_predictions(0.92, 0.04))
        self.assertEqual(len(result), 1)

    def test_two_predictions_cross_target(self) -> None:
        result = filter_predictions_by_confidence(_predictions(0.75, 0.16, 0.05))
        self.assertEqual(len(result), 2)

    def test_three_predictions_cross_target(self) -> None:
        result = filter_predictions_by_confidence(_predictions(0.65, 0.20, 0.08, 0.04))
        self.assertEqual(len(result), 3)

    def test_exact_cumulative_target_stops_selection(self) -> None:
        result = filter_predictions_by_confidence(_predictions(0.55, 0.20, 0.15, 0.05))
        self.assertEqual(len(result), 3)

    def test_five_predictions_are_required_to_cross_target(self) -> None:
        result = filter_predictions_by_confidence(
            _predictions(0.56, 0.12, 0.09, 0.08, 0.06)
        )
        self.assertEqual(len(result), 5)

    def test_five_predictions_return_when_total_is_below_target(self) -> None:
        result = filter_predictions_by_confidence(
            _predictions(0.55, 0.10, 0.08, 0.07, 0.05)
        )
        self.assertEqual(len(result), 5)

    def test_selection_never_returns_more_than_five_predictions(self) -> None:
        result = filter_predictions_by_confidence(
            _predictions(0.55, 0.05, 0.05, 0.05, 0.05, 0.05)
        )
        self.assertEqual(len(result), 5)

    def test_all_response_lengths_are_supported(self) -> None:
        score_rows = (
            (0.50,),
            (0.91, 0.04),
            (0.70, 0.21, 0.04),
            (0.65, 0.15, 0.11, 0.04),
            (0.55, 0.14, 0.11, 0.10, 0.04),
            (0.51, 0.12, 0.10, 0.09, 0.08),
        )
        lengths = {
            len(filter_predictions_by_confidence(_predictions(*scores)))
            for scores in score_rows
        }
        self.assertEqual(lengths, {0, 1, 2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()
