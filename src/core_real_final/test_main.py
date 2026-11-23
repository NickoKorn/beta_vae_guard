"""Load the five inputs used by the Varix entry point.

Run from the repository root: python -m unittest -v src.core_real_final.test_main
"""

import json
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from src.core_real_final.load_data import gudhi, tcga
from src.core_real_final.main import DEFAULT_CONFIG, ROOT


CONFIG = json.loads(DEFAULT_CONFIG.read_text())


class MainInputTests(unittest.TestCase):
    def test_tcga_groups(self):
        settings = CONFIG["data"]["tcga"]
        for group in CONFIG["experiment"]["groups"]:
            with self.subTest(group=group):
                modalities, annotation = tcga(group, ROOT / settings["root"], settings)
                self.assertEqual(set(modalities), {"RNA", "METH"})
                self.assertTrue(annotation.index.is_unique)
                for frame in modalities.values():
                    self.assertTrue(frame.index.equals(annotation.index))
                    self.assertGreater(frame.shape[1], 0)
                base = group.removesuffix(settings["lowvar_suffix"])
                labels = settings["groups"][base]
                if labels is not None:
                    self.assertEqual(set(annotation[settings["label"]]), set(labels))
                print(f"TCGA {group}: {len(annotation)} samples", flush=True)
                del modalities, annotation, frame

    def test_gudhi_topologies(self):
        settings = CONFIG["data"]["gudhi"]
        with TemporaryDirectory() as cache:
            for topology in CONFIG["experiment"]["topologies"]:
                dimensions = {"circle": 2, "sphere": 3}[topology]
                with self.subTest(topology=topology):
                    modalities, annotation = gudhi(topology, cache, settings)
                    self.assertEqual(set(modalities), {topology})
                    points = modalities[topology]
                    self.assertEqual(points.shape, (settings["n_samples"], dimensions))
                    self.assertTrue(points.index.is_unique and points.index.equals(annotation.index))
                    np.testing.assert_allclose(np.linalg.norm(points, axis=1), settings.get("radius", 1.0))
                    print(f"GUDHI {topology}: {len(points)} samples", flush=True)


if __name__ == "__main__":
    unittest.main()
