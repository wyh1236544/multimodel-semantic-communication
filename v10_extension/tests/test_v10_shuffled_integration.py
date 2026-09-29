"""Synthetic PyTorch check for the corrected single-image Common control."""
import unittest
import torch
from test_v10_integration import install_stubs, tiny_config


class ShuffledCommonIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = install_stubs()
        cls.stub.start()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        cls.stub.stop()

    def test_single_image_requires_a_real_donor(self):
        from models.joint_image_text_model import JointImageTextSemanticCommunicationModel
        from v10.model import V10Model
        torch.manual_seed(42)
        config = tiny_config()
        model = V10Model(JointImageTextSemanticCommunicationModel(config['model']),
                         config['v10']).eval()
        images = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            donor = model(images[1:2], ['a different image'], 'awgn', 20.,
                          'common_only', common_task_only=True)['z_transmitted'].detach()
            with self.assertRaisesRegex(ValueError, 'batch size one'):
                model(images[:1], ['the target image'], 'awgn', 20.,
                      'common_only_shuffled')
            shuffled = model(images[:1], ['the target image'], 'awgn', 20.,
                             'common_only_shuffled', shuffled_common_payload=donor)
        self.assertTrue(torch.equal(shuffled['z_transmitted'], donor))
        self.assertFalse(torch.equal(shuffled['z_transmitted'], shuffled['z_common']))
        self.assertEqual(int(shuffled['link_stats']['forward_uses']), 352)


if __name__ == '__main__':
    unittest.main()
