#!/usr/bin/env python

import unittest
import os
import tempfile

from hamoco import ClassificationModel
import numpy

class Test(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
    def test_model_building(self):

        # Create model and read data files
        model = ClassificationModel()
        model.read_dataset(self.data_dir)
        self.assertEqual(set(model.classes), set([0,1,2,3,4,5]), 'not the expected labels')

        # Process the data
        model.process_dataset()
        
        # Train the model
        model.train(hidden_layers=(5,5,5), epochs=5)
        model.save_model(os.path.join(self.data_dir, 'phony_model.h5'))

    def test_read_dataset_ignores_macos_resource_forks(self):
        with tempfile.TemporaryDirectory() as data_dir:
            sample_path = os.path.join(data_dir, 'sample.dat')
            resource_fork_path = os.path.join(data_dir, '._sample.dat')
            features = ' '.join(['0.0'] * ClassificationModel.n_features)
            with open(sample_path, 'w') as sample:
                sample.write(f'0\n{features}\n')
            with open(resource_fork_path, 'wb') as resource_fork:
                resource_fork.write(b'\x00\x05\x16\x07binary resource fork')

            model = ClassificationModel()
            model.read_dataset(data_dir)

        self.assertEqual(model.n_samples, 1)
        numpy.testing.assert_allclose(model.data[0], numpy.zeros(ClassificationModel.n_features))
        
if __name__ == '__main':
    unittest.main()
