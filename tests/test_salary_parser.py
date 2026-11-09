"""Unit tests for salary parsing and validation."""
import unittest
from src.data.salary_parser import parse_salary_string

class TestSalaryParser(unittest.TestCase):
    
    def test_range_salary_vnd(self):
        """Test parsing standard VND ranges."""
        min_v, max_v, mean_v, cat, status = parse_salary_string("15 - 25 triệu")
        self.assertEqual(status, "valid")
        self.assertEqual(cat, "range")
        self.assertEqual(min_v, 15_000_000.0)
        self.assertEqual(max_v, 25_000_000.0)
        self.assertEqual(mean_v, 20_000_000.0)
        
    def test_range_salary_full_digits(self):
        """Test parsing full numeric digits with thousand separators."""
        min_v, max_v, mean_v, cat, status = parse_salary_string("12.000.000 - 18.000.000 VND")
        self.assertEqual(status, "valid")
        self.assertEqual(cat, "range")
        self.assertEqual(min_v, 12_000_000.0)
        self.assertEqual(max_v, 18_000_000.0)
        self.assertEqual(mean_v, 15_000_000.0)

    def test_exact_salary(self):
        """Test single exact salary."""
        min_v, max_v, mean_v, cat, status = parse_salary_string("20 triệu")
        self.assertEqual(status, "valid")
        self.assertEqual(cat, "exact")
        self.assertEqual(mean_v, 20_000_000.0)

    def test_upper_bound_only(self):
        """Test 'up to / lên đến' one-sided salary disclosures."""
        min_v, max_v, mean_v, cat, status = parse_salary_string("Lên đến 35 triệu")
        self.assertEqual(status, "valid")
        self.assertEqual(cat, "upper_bound_only")
        self.assertIsNone(min_v)
        self.assertEqual(max_v, 35_000_000.0)
        self.assertEqual(mean_v, 35_000_000.0)

    def test_lower_bound_only(self):
        """Test 'từ / from' one-sided salary disclosures."""
        min_v, max_v, mean_v, cat, status = parse_salary_string("Từ 15 triệu")
        self.assertEqual(status, "valid")
        self.assertEqual(cat, "lower_bound_only")
        self.assertEqual(min_v, 15_000_000.0)
        self.assertIsNone(max_v)
        self.assertEqual(mean_v, 15_000_000.0)

    def test_usd_conversion(self):
        """Test USD currency conversion."""
        min_v, max_v, mean_v, cat, status = parse_salary_string("$1000 - $2000")
        self.assertEqual(status, "valid")
        self.assertEqual(cat, "range")
        self.assertEqual(min_v, 25_000_000.0)
        self.assertEqual(max_v, 50_000_000.0)
        self.assertEqual(mean_v, 37_500_000.0)

    def test_unspecified_salaries(self):
        """Test negotiable / unspecified keywords."""
        for text in ["Thỏa thuận", "Thoả thuận", "Cạnh tranh", "Negotiable", "Hấp dẫn"]:
            with self.subTest(text=text):
                _, _, _, cat, status = parse_salary_string(text)
                self.assertEqual(cat, "unspecified")
                self.assertEqual(status, "unspecified")

    def test_implausible_outliers(self):
        """Test filtering of implausibly low or high salaries."""
        # Too low (< 3M)
        _, _, _, cat_low, status_low = parse_salary_string("500.000 - 1.500.000")
        self.assertEqual(status_low, "implausible")
        
        # Too high (> 150M)
        _, _, _, cat_high, status_high = parse_salary_string("200 - 300 triệu")
        self.assertEqual(status_high, "implausible")

if __name__ == '__main__':
    unittest.main()

