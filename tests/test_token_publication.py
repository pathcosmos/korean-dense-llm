from pathlib import Path
import tempfile
import unittest
from scripts.benchmark_next_stage import publish_tree


class TokenPublicationTest(unittest.TestCase):
    def test_copy_resume_and_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source';src.mkdir();(src/'nested').mkdir()
            (src/'nested/a.bin').write_bytes(bytes(range(256))*100)
            (src/'report.json').write_text('{"status":"fixture"}')
            target=root/'published';publish_tree(src,target)
            self.assertEqual((src/'nested/a.bin').read_bytes(),(target/'nested/a.bin').read_bytes())
            target.rename(root/'published.publishing')
            publish_tree(src,target)
            self.assertTrue(target.is_dir())
            (src/'nested/a.bin').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'input changed'):publish_tree(src,target)

    def test_partial_staging_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source';src.mkdir();(src/'x').write_bytes(b'original')
            stage=root/'target.publishing';stage.mkdir();(stage/'x').write_bytes(b'partial')
            with self.assertRaises(FileNotFoundError):publish_tree(src,root/'target')
            self.assertEqual((stage/'x').read_bytes(),b'partial');self.assertFalse((root/'target').exists())
