from pathlib import Path
from utils import DatasetFolder, Dataloader
import unittest

class TestDatasetFolder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_path = "images"
        cls.datasetFolder = DatasetFolder(cls.default_path)

    def test_create(self):
        self.assertEqual(self.datasetFolder.root, Path("images"), "DatasetFolder has incorrect root")

    def test_invalid_root(self):
        with self.assertRaises(FileNotFoundError):
            DatasetFolder("this is not a root")

class TestDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_path = "images"
        cls.default_batchsize = 2
        cls.datasetFolder = DatasetFolder(cls.default_path)
        cls.dataLoader = Dataloader(cls.datasetFolder, cls.default_batchsize)

    def test_create(self):
        self.assertEqual(self.dataLoader.batch_size, self.default_batchsize, "dataloader has incorrect batch_size")

if __name__ == "__main__":
    unittest.main()
