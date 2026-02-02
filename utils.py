try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def to_torch(v):
    """Convert Python list/float to PyTorch tensor"""
    if not HAS_TORCH:
        return v
    if isinstance(v, torch.Tensor):
        return v
    if is_scalar(v):
        return torch.tensor(v, requires_grad=True, dtype=torch.float32)
    # Convert list to tensor
    return torch.tensor(v, requires_grad=True, dtype=torch.float32)

def from_torch(v):
    """Convert PyTorch tensor to Python list/float for display"""
    if not HAS_TORCH or not isinstance(v, torch.Tensor):
        # Handle Python complex numbers
        if isinstance(v, complex):
            if abs(v.imag) < 1e-10:
                return v.real
            return v
        return v
    if v.numel() == 1:
        val = v.item()
        # Extract real part if imaginary is negligible
        if isinstance(val, complex) and abs(val.imag) < 1e-10:
            return val.real
        return val
    return v.detach().tolist()

def is_torch_tensor(v):
    """Check if value is a torch tensor"""
    return HAS_TORCH and isinstance(v, torch.Tensor)

def is_scalar(v):
    return isinstance(v, float) or (is_torch_tensor(v) and v.numel() == 1)

def is_vector(v):
    if is_torch_tensor(v):
        return v.dim() == 1
    return isinstance(v, list) and len(v) > 0 and all(is_scalar(x) or isinstance(x, float) for x in v)

def is_matrix(v):
    """Check if v is a nested list (matrix/tensor)"""
    if is_torch_tensor(v):
        return v.dim() >= 2
    if not isinstance(v, list) or len(v) == 0:
        return False
    # Check if first element is a list (not scalar)
    return isinstance(v[0], list)

def get_shape(v):
    """Recursively get the shape of a nested list or torch tensor"""
    if is_torch_tensor(v):
        return tuple(v.shape)
    if is_scalar(v):
        return ()
    if not isinstance(v, list):
        return ()
    shape = [len(v)]
    if len(v) > 0 and isinstance(v[0], list):
        inner_shape = get_shape(v[0])
        shape.extend(inner_shape)
    return tuple(shape)

def is_function(v):
    return isinstance(v, dict) and "params" in v and "body" in v

def infer_type(v):
    # Check for complex numbers
    if isinstance(v, complex):
        if v.imag == 0:
            return "ℝ"  # Real result from complex calculation
        return "ℂ"
    if is_torch_tensor(v) and v.is_complex():
        if v.imag.abs().max() < 1e-10:
            return "ℝ"  # Effectively real
        return "ℂ"
    if is_scalar(v):
        return "ℝ"
    if is_vector(v):
        return f"ℝ[{len(v)}]"
    if is_matrix(v):
        shape = get_shape(v)
        dims = ",".join(str(d) for d in shape)
        return f"ℝ[{dims}]"
    raise TypeError("Unknown type")

def format_tensor_type(tensor_spec):
    """Format tensor type with variance annotations"""
    if isinstance(tensor_spec, tuple) and tensor_spec[0] == "tensor":
        dims = tensor_spec[1]
        parts = []
        for dim, variance in dims:
            if variance == "contravariant":
                parts.append(f"+{dim}")
            elif variance == "covariant":
                parts.append(f"-{dim}")
            else:  # invariant
                parts.append(str(dim))
        return f"ℝ[{','.join(parts)}]"
    return str(tensor_spec)

def flatten(lst):
    """Flatten a nested list structure"""
    if is_scalar(lst):
        return [lst]
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def reshape(flat_list, shape):
    """Reshape a flat list into nested structure according to shape"""
    if len(shape) == 0:
        return flat_list[0]
    if len(shape) == 1:
        return flat_list[:shape[0]]
    
    size = shape[0]
    inner_size = 1
    for dim in shape[1:]:
        inner_size *= dim
    
    result = []
    for i in range(size):
        start = i * inner_size
        end = start + inner_size
        result.append(reshape(flat_list[start:end], shape[1:]))
    return result
