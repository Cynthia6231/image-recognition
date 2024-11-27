try:
    #较新的PyTorch版本中推荐使用的加载预训练权重的方法
    from torch.hub import load_state_dict_from_url
except ImportError:
    #PyTorch版本较旧
    from torch.utils.model_zoo import load_url as load_state_dict_from_url