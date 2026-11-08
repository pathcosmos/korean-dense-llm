"""Regression: meta/to_empty must restore RMSNorm's constant initialization."""
import unittest
import torch
from model.model import ModelConfig, Transformer, RMSNorm


class InitializationTest(unittest.TestCase):
    def test_meta_materialization_initializes_norms_and_gradients(self):
        cfg = ModelConfig(vocab_size=256, n_layers=1, dim=32, n_heads=4,
                          n_kv_heads=2, ffn_dim=64, max_seq_len=16)
        with torch.device('meta'):
            model = Transformer(cfg)
        model.to_empty(device='cpu')
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, RMSNorm):
                    module.weight.fill_(float('nan'))
            model.init_weights()
        loss = model(torch.randint(256, (2, 16)), torch.randint(256, (2, 16)))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
        for module in model.modules():
            if isinstance(module, RMSNorm):
                self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))


if __name__ == '__main__':
    unittest.main()
