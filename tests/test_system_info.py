import unittest

from scripts.system_info import GIB, format_bytes, recommend_torch_device


class FormatBytesTests(unittest.TestCase):
    def test_formats_gib(self):
        self.assertEqual(format_bytes(2 * GIB), "2.00 GiB")

    def test_handles_unknown_memory(self):
        self.assertEqual(format_bytes(None), "unavailable")


class RecommendTorchDeviceTests(unittest.TestCase):
    def test_prefers_cuda_when_available(self):
        self.assertEqual(
            recommend_torch_device(cuda_available=True, mps_available=True),
            "cuda:0",
        )

    def test_uses_mps_when_cuda_is_unavailable(self):
        self.assertEqual(
            recommend_torch_device(cuda_available=False, mps_available=True),
            "mps",
        )

    def test_falls_back_to_cpu(self):
        self.assertEqual(
            recommend_torch_device(cuda_available=False, mps_available=False),
            "cpu",
        )


if __name__ == "__main__":
    unittest.main()
