"""Unit tests for location normalization logic."""
import unittest
from src.data.cleaner import normalize_location

class TestLocationNormalization(unittest.TestCase):
    
    def test_hanoi_with_quan(self):
        """Verify that Hanoi locations containing the keyword 'quận' do NOT map to Ho Chi Minh."""
        cases = [
            ("Quận Cầu Giấy, Hà Nội", "Hà Nội"),
            ("Quận Thanh Xuân, Thành phố Hà Nội", "Hà Nội"),
            ("Quận Hoàn Kiếm, Hà Nội", "Hà Nội"),
            ("Phường Dịch Vọng Hậu, Quận Cầu Giấy, Thành phố Hà Nội, Việt Nam", "Hà Nội"),
            ("Số 275 Nguyễn Trãi, Quận Thanh Xuân, TP Hà nội", "Hà Nội"),
            ("Quận Nam Từ Liêm, Hà Nội", "Hà Nội"),
            ("Quận Ba Đình, TP Hà Nội", "Hà Nội"),
            ("Quận Đống Đa, Hà Nội", "Hà Nội"),
            ("quận Hai Bà Trưng, Hà Nội", "Hà Nội"),
            ("Quận Hà Đông, Hà Nội", "Hà Nội")
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_location(raw), expected)

    def test_hcm_locations(self):
        """Verify proper mapping of Ho Chi Minh locations."""
        cases = [
            ("Quận 1, Hồ Chí Minh", "Hồ Chí Minh"),
            ("Thành phố Hồ Chí Minh", "Hồ Chí Minh"),
            ("Quận Bình Thạnh, TP.HCM", "Hồ Chí Minh"),
            ("Thủ Đức, Hồ Chí Minh", "Hồ Chí Minh"),
            ("Quận Tân Bình, TP HCM", "Hồ Chí Minh"),
            ("Quận 7, TP. Hồ Chí Minh", "Hồ Chí Minh"),
            ("Hồ Chí Minh", "Hồ Chí Minh"),
            ("Saigon, Vietnam", "Hồ Chí Minh")
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_location(raw), expected)

    def test_danang_locations(self):
        """Verify Da Nang location mapping."""
        cases = [
            ("Đà Nẵng", "Đà Nẵng"),
            ("Quận Hải Châu, Đà Nẵng", "Đà Nẵng"),
            ("Quận Sơn Trà, TP Đà Nẵng", "Đà Nẵng"),
            ("Da Nang", "Đà Nẵng")
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_location(raw), expected)

    def test_multiple_locations(self):
        """Verify multi-city job postings."""
        cases = [
            ("Hà Nội, Hồ Chí Minh", "Multiple"),
            ("Quận 7, Hồ Chí Minh; Cầu Giấy, Hà Nội", "Multiple"),
            ("Hà Nội / TP HCM / Đà Nẵng", "Multiple")
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_location(raw), expected)

    def test_remote_and_other(self):
        """Verify remote and other/unknown locations."""
        self.assertEqual(normalize_location("Remote - Làm việc từ xa"), "Remote")
        self.assertEqual(normalize_location("Work from home"), "Remote")
        self.assertEqual(normalize_location("Bình Dương"), "Khác")
        self.assertEqual(normalize_location("Cần Thơ"), "Khác")
        self.assertEqual(normalize_location(None), "Khác")
        self.assertEqual(normalize_location(""), "Khác")

if __name__ == '__main__':
    unittest.main()

