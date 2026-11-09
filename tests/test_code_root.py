"""PROJECT_ROOT may be an output tree outside the checkout: code paths must still resolve to the running code."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import data_contracts as dc


class CodeRootTest(unittest.TestCase):
    def test_code_resolves_to_checkout_when_project_root_is_elsewhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(dc, 'PROJECT_ROOT', tmp):
                for rel in ('data/source_audit.py', 'common/data_contracts.py', 'tokenizer/train_tokenizer.py'):
                    self.assertEqual(dc.resolve(rel), dc.CODE_ROOT / rel)
                    self.assertTrue(dc.sha256_file(dc.resolve(rel)))
                # data stays under PROJECT_ROOT
                self.assertEqual(dc.resolve('data/clean/wiki/a.parquet'), Path(tmp) / 'data/clean/wiki/a.parquet')
                self.assertEqual(dc.resolve('runs/x.json'), Path(tmp) / 'runs/x.json')
                self.assertEqual(dc.resolve('requirements.txt'), Path(tmp) / 'requirements.txt')   # only .py falls back

    def test_copy_under_project_root_wins_and_missing_files_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            own = Path(tmp) / 'data' / 'source_audit.py'
            own.parent.mkdir(parents=True)
            own.write_text('# local copy')
            with patch.object(dc, 'PROJECT_ROOT', tmp):
                self.assertEqual(dc.resolve('data/source_audit.py'), own)
                missing = dc.resolve('data/no_such_module.py')          # absent in both -> original path, so errors name it
                self.assertEqual(missing, Path(tmp) / 'data/no_such_module.py')
                self.assertFalse(missing.exists())
                absolute = Path(tmp) / 'abs.py'
                self.assertEqual(dc.resolve(str(absolute)), absolute)

    def test_default_root_is_unchanged(self):
        self.assertEqual(Path(dc.PROJECT_ROOT).resolve(), dc.CODE_ROOT)
        self.assertEqual(dc.resolve('data/filters.py'), Path(dc.PROJECT_ROOT) / 'data/filters.py')

    def test_audit_style_code_hashing_works_outside_the_checkout(self):
        files = ['data/source_audit.py', 'common/source_inventory.py', 'common/data_contracts.py', 'data/probe_code_math_overlap.py']
        with tempfile.TemporaryDirectory() as tmp, patch.object(dc, 'PROJECT_ROOT', tmp):
            hashes = {p: dc.sha256_file(dc.resolve(p)) for p in files}     # the line that raised FileNotFoundError in a clean clone
        with patch.object(dc, 'PROJECT_ROOT', str(dc.CODE_ROOT)):
            self.assertEqual(hashes, {p: dc.sha256_file(dc.resolve(p)) for p in files})


if __name__ == '__main__':
    unittest.main()
