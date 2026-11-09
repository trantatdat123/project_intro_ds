"""Unit tests for deduplication module."""
import unittest
import pandas as pd
from src.data.loader import deduplicate_merged

class TestDeduplication(unittest.TestCase):
    
    def test_exact_and_normalized_duplicates(self):
        """Test that casing, extra spaces, and corporate suffixes are handled during deduplication."""
        data = [
            {
                "unique_job_id": "job_1",
                "source_id": "1",
                "source": "tinixai",
                "job_title": "[Hà Nội] Senior Python Developer",
                "company_name": "Công ty Cổ phần Công nghệ ABC",
                "job_description": "Chúng tôi cần tuyển Senior Python Developer có kinh nghiệm 3 năm với Django và FastAPI.",
                "location": "Hà Nội",
                "year": 2025
            },
            {
                "unique_job_id": "job_2",
                "source_id": "2",
                "source": "tinixai",
                "job_title": "senior python developer",
                "company_name": "Công ty ABC",
                "job_description": "Chúng tôi cần tuyển Senior Python Developer có kinh nghiệm 3 năm với Django và FastAPI.",
                "location": "Hà Nội",
                "year": 2025
            },
            {
                "unique_job_id": "job_3",
                "source_id": "3",
                "source": "topcv",
                "job_title": "Junior Java Developer",
                "company_name": "Công ty TNHH Phần mềm XYZ",
                "job_description": "Tuyển dụng lập trình viên Java Spring Boot mới tốt nghiệp hoặc 1 năm kinh nghiệm.",
                "location": "Hồ Chí Minh",
                "year": 2026
            }
        ]
        df = pd.DataFrame(data)
        deduped, removed, audit = deduplicate_merged(df)
        
        self.assertEqual(len(deduped), 2)
        self.assertEqual(len(removed), 1)
        self.assertEqual(deduped.iloc[0]['unique_job_id'], "job_1")
        self.assertEqual(deduped.iloc[1]['unique_job_id'], "job_3")

if __name__ == '__main__':
    unittest.main()

