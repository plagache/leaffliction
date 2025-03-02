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

    def test_same_name_directory(self):
        """
            Where do we raise the error of category with the same name, but path is different

            directory/
            ├── class_x
            │   ├── x_sample_1.jpg
            │   ├── x_sample_2.jpg
            │   └── sub_class_x
            │       └── sub_class_x_sample_1.jpg
            └── class_y
                ├── y_sample_2.jpg
                ├── y_sample_2.jpg
                └── class_x
                    └── other.jpg
        """
        return

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
