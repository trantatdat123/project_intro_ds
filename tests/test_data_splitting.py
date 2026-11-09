"""Unit tests for data splitting policy (80/10/10 with zero overlap)."""
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class TestDataSplitting(unittest.TestCase):
    
    def test_eighty_ten_ten_split_proportions(self):
        """Verify that 80/10/10 train/val/test split produces correct proportions."""
        n_samples = 1000
        indices = np.arange(n_samples)
        
        # Step 1: 80% train, 20% temp
        train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=42)
        # Step 2: 50/50 split of temp -> 10% val, 10% test
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)
        
        self.assertEqual(len(train_idx), 800)
        self.assertEqual(len(val_idx), 100)
        self.assertEqual(len(test_idx), 100)
        
    def test_zero_leakage_overlap(self):
        """Verify that train, val, and test sets are mutually disjoint (zero leakage)."""
        n_samples = 500
        indices = np.arange(n_samples)
        
        train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)
        
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)
        
        self.assertTrue(train_set.isdisjoint(val_set))
        self.assertTrue(train_set.isdisjoint(test_set))
        self.assertTrue(val_set.isdisjoint(test_set))
        self.assertEqual(len(train_set) + len(val_set) + len(test_set), n_samples)

    def test_stratification_preservation(self):
        """Verify that stratification maintains proportional representation across splits."""
        labels = np.array(['ClassA'] * 700 + ['ClassB'] * 200 + ['ClassC'] * 100)
        indices = np.arange(len(labels))
        
        train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
        temp_labels = labels[temp_idx]
        val_rel, test_rel = train_test_split(np.arange(len(temp_idx)), test_size=0.50, random_state=42, stratify=temp_labels)
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]
        
        train_counts = pd.Series(labels[train_idx]).value_counts(normalize=True)
        val_counts = pd.Series(labels[val_idx]).value_counts(normalize=True)
        test_counts = pd.Series(labels[test_idx]).value_counts(normalize=True)
        
        np.testing.assert_almost_equal(train_counts['ClassA'], 0.70, decimal=2)
        np.testing.assert_almost_equal(val_counts['ClassA'], 0.70, decimal=2)
        np.testing.assert_almost_equal(test_counts['ClassA'], 0.70, decimal=2)

if __name__ == '__main__':
    unittest.main()

