import unittest
from data.probe_url_overlap import canonical_url


class UrlNormalizationTest(unittest.TestCase):
    def test_remove_fragment_and_default_port(self):
        self.assertEqual(canonical_url(' HTTPS://Example.COM:443/a?q=1#part '), 'https://example.com/a?q=1')
        self.assertEqual(canonical_url('http://example.com'), 'http://example.com/')

    def test_preserve_distinct_resources(self):
        self.assertNotEqual(canonical_url('http://example.com/a'), canonical_url('https://example.com/a'))
        self.assertNotEqual(canonical_url('https://example.com/a?q=1'), canonical_url('https://example.com/a?q=2'))

    def test_invalid_values(self):
        for x in (None, '', 'bad url', 'https://[bad', 'http://example.com:bad/a'):
            self.assertIsNone(canonical_url(x))
