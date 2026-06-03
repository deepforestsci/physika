from physika.utils.print_utils import _infer_type
import torch


class TestInferType:
    """Tests for ``_infer_type`` function"""

    def testRealValues(self):
        """Tests for Real ``ℝ`` values"""
        scalar_R = _infer_type(3.0)
        assert scalar_R == 'ℝ'

        tensor_R = _infer_type(torch.tensor([1.0, 2.0, 3.0]))
        assert tensor_R == 'ℝ[3]'

        tensor_R_2D = _infer_type(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert tensor_R_2D == 'ℝ[2,2]'

    def testComplexValues(self):
        """Tests for complex ``ℂ`` values"""
        complex_R = _infer_type(3j)
        assert complex_R == 'ℂ'

        complex_R = _infer_type(torch.tensor([1j, 2j, 3j]))
        assert complex_R == 'ℂ[3]'

        complex_R_2D = _infer_type(torch.tensor([[1j, 2j], [3j, 4j]]))
        assert complex_R_2D == 'ℂ[2,2]'
