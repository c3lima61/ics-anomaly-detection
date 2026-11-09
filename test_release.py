"""Check provenance portability without loading a model or changing research files."""
from pathlib import Path
import tempfile
import unittest

from evaluate_attacks import frozen_hashes
from evaluate_defence import stage4_hashes


class PortableProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (Path(__file__).resolve().parent / ".validation").mkdir(exist_ok=True)

    def test_asset_hashes_survive_relocation_and_detect_changes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent / '.validation') as root:
            copies = []
            for name in ('first', 'second'):
                base = Path(root) / name
                stage3 = base / 'stage3'
                stage3.mkdir(parents=True)
                data = base / 'data.csv'
                data.write_text('synthetic data')
                (stage3 / 'model').write_bytes(b'model fixture; never deserialized')
                copies.append((data, stage3))
            self.assertEqual(frozen_hashes(*copies[0]), frozen_hashes(*copies[1]))
            (copies[1][1] / 'model').write_bytes(b'changed')
            self.assertNotEqual(frozen_hashes(*copies[0]), frozen_hashes(*copies[1]))

    def test_stage4_keys_have_no_machine_path(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent / '.validation') as root:
            directory = Path(root)
            (directory / 'archive.csv').write_text('synthetic data')
            self.assertEqual(list(stage4_hashes(directory)), ['stage4/archive.csv'])


if __name__ == '__main__':
    unittest.main()
