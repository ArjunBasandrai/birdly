from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from birdly.inference.predict import (
    DEFAULT_TAXON_NAMES_PATH,
    Predictor,
    _format_predictions,
    load_predictor,
)
from scripts.generate_taxon_names import (
    DEFAULT_USER_AGENT,
    fetch_taxon_names,
    generate_taxon_names,
    parse_taxa_response,
)
from scripts.prepare_cloud_run import SOURCE_FILES, _validate_taxon_names


class TaxonNameGenerationTests(unittest.TestCase):
    def test_response_uses_common_and_scientific_names(self) -> None:
        result = parse_taxa_response(
            {
                "results": [
                    {
                        "id": 145310,
                        "name": "Spinus tristis",
                        "preferred_common_name": "American Goldfinch",
                    }
                ]
            },
            [145310],
        )
        self.assertEqual(
            result["145310"],
            {
                "common_name": "American Goldfinch",
                "scientific_name": "Spinus tristis",
            },
        )

    def test_missing_common_name_uses_scientific_name(self) -> None:
        result = parse_taxa_response(
            {"results": [{"id": 7, "name": "Example scientificus"}]},
            [7],
        )
        self.assertEqual(result["7"]["common_name"], "Example scientificus")
        self.assertEqual(result["7"]["scientific_name"], "Example scientificus")

    def test_complete_existing_mapping_does_not_call_api_fetcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            class_mapping_path = root / "class_mapping.json"
            output_path = root / "taxon_names.json"
            class_mapping_path.write_text(
                json.dumps({"class_count": 1, "class_to_taxon": {"0": 7}}),
                encoding="utf-8",
            )
            expected = {
                "7": {
                    "common_name": "Example Bird",
                    "scientific_name": "Example scientificus",
                }
            }
            output_path.write_text(json.dumps(expected), encoding="utf-8")
            fetcher = mock.Mock(
                side_effect=AssertionError("API fetch is not permitted")
            )

            actual, reused = generate_taxon_names(
                class_mapping_path,
                output_path,
                fetcher=fetcher,
            )

        self.assertTrue(reused)
        self.assertEqual(actual, expected)
        fetcher.assert_not_called()

    def test_api_request_uses_english_and_new_york_preferences(self) -> None:
        response = {
            "results": [
                {
                    "id": 145310,
                    "name": "Spinus tristis",
                    "preferred_common_name": "American Goldfinch",
                }
            ]
        }
        with mock.patch(
            "scripts.generate_taxon_names._fetch_json",
            return_value=response,
        ) as fetch_json:
            result = fetch_taxon_names([145310], request_interval_seconds=0)

        self.assertIn("locale=en", fetch_json.call_args.args[0])
        self.assertIn("preferred_place_id=48", fetch_json.call_args.args[0])
        self.assertEqual(fetch_json.call_args.kwargs["user_agent"], DEFAULT_USER_AGENT)
        self.assertEqual(result["145310"]["common_name"], "American Goldfinch")


class PredictionNameTests(unittest.TestCase):
    def setUp(self) -> None:
        names = json.loads(DEFAULT_TAXON_NAMES_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            names["145310"],
            {
                "common_name": "American Goldfinch",
                "scientific_name": "Spinus tristis",
            },
        )
        self.predictor = Predictor(
            model=lambda value: value,
            transform=lambda value: value,
            class_to_taxon={0: 145310, 1: 473},
            taxon_to_names={
                145310: names["145310"],
                473: names["473"],
            },
            device="cpu",
            model_name="test",
            backend="test",
            precision="float32",
            compile_mode=None,
            autocast_dtype=None,
            image_size=224,
            preprocessing={},
        )

    def test_prediction_names_ids_and_scores_are_preserved(self) -> None:
        scores = [0.6, 0.3]
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("Prediction must not use iNaturalist"),
        ):
            predictions = _format_predictions(self.predictor, [0, 1], scores)

        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0]["taxon_id"], 145310)
        self.assertEqual(predictions[0]["common_name"], "American Goldfinch")
        self.assertEqual(predictions[0]["scientific_name"], "Spinus tristis")
        self.assertEqual([prediction["score"] for prediction in predictions], scores)

    def test_predictor_loads_static_names_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            class_mapping_path = root / "class_mapping.json"
            preprocessing_path = root / "preprocessing.json"
            taxon_names_path = root / "taxon_names.json"
            model_path = root / "birdly-fp32.onnx"
            metadata_path = root / "metadata.json"
            class_mapping_path.write_text(
                json.dumps({"class_count": 1, "class_to_taxon": {"0": 145310}}),
                encoding="utf-8",
            )
            preprocessing_path.write_text(
                json.dumps(
                    {
                        "source_checkpoint_sha256": "checkpoint-hash",
                        "image_size": 224,
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    }
                ),
                encoding="utf-8",
            )
            taxon_names_path.write_text(
                json.dumps(
                    {
                        "145310": {
                            "common_name": "American Goldfinch",
                            "scientific_name": "Spinus tristis",
                        }
                    }
                ),
                encoding="utf-8",
            )
            model_path.write_bytes(b"test model")
            metadata_path.write_text(
                json.dumps(
                    {
                        "source_checkpoint_sha256": "checkpoint-hash",
                        "class_mapping_sha256": hashlib.sha256(
                            class_mapping_path.read_bytes()
                        ).hexdigest(),
                        "preprocessing_sha256": hashlib.sha256(
                            preprocessing_path.read_bytes()
                        ).hexdigest(),
                        "class_count": 1,
                        "model_name": "test-model",
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch(
                    "birdly.inference.predict.load_onnx_runtime_model",
                    return_value=lambda value: value,
                ),
                mock.patch(
                    "urllib.request.urlopen",
                    side_effect=AssertionError("Inference must not use iNaturalist"),
                ),
            ):
                predictor = load_predictor(
                    backend="onnxruntime",
                    class_mapping_path=class_mapping_path,
                    taxon_names_path=taxon_names_path,
                    preprocessing_path=preprocessing_path,
                    onnx_model_path=model_path,
                    onnx_metadata_path=metadata_path,
                )

        self.assertEqual(
            predictor.taxon_to_names[145310],
            {
                "common_name": "American Goldfinch",
                "scientific_name": "Spinus tristis",
            },
        )


class ProductionBundleNameTests(unittest.TestCase):
    def test_cloud_run_source_and_validation_include_taxon_names(self) -> None:
        self.assertIn("src/birdly/inference/taxon_names.json", SOURCE_FILES)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "taxon_names.json"
            path.write_text(
                json.dumps(
                    {
                        "7": {
                            "common_name": "Example Bird",
                            "scientific_name": "Example scientificus",
                        }
                    }
                ),
                encoding="utf-8",
            )
            count = _validate_taxon_names(
                {"class_count": 1, "class_to_taxon": {"0": 7}},
                path,
            )
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
