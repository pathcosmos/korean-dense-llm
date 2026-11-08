import unittest
from data.filters import check_en, check_math


class MathFilterTest(unittest.TestCase):
    def test_preserve_math_heavy_text(self):
        text = 'We study the following distinct equations: ' + ' '.join(
            f'${i} = ({i}+1) / ({i}+2)$' for i in range(30))
        self.assertEqual(check_en(text), 'low_alnum')
        self.assertIsNone(check_math(text))

    def test_do_not_relax_other_filters(self):
        self.assertEqual(check_math('$x=1$'), 'short')
        self.assertEqual(check_math('explanation ' + '!' * 500), 'low_alnum')
        self.assertEqual(check_math(('Repeated mathematical line $x=1$\n') * 20), 'dup_lines')
