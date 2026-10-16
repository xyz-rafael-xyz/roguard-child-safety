import unittest
from training.archive_instruction_adapter_v8 import validate, FORMAT


class InstructionArchiveTests(unittest.TestCase):
    def test_incomplete_or_foreign_architecture_is_rejected(self):
        spec={'format':FORMAT,'base_model':'mlx-community/Qwen3-4B-Instruct-2507-4bit',
              'adapter_config':{'fine_tune_type':'lora','num_layers':8,'lora_parameters':{'rank':8,'dropout':0.0,'scale':16.0}},'tensors':{}}
        with self.assertRaises(ValueError):validate(spec)
        spec['adapter_config']['num_layers']=9
        with self.assertRaises(ValueError):validate(spec)

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(ValueError):validate({'format':'pickle'})
