import torch
import torch.nn as nn

class GlobalFeatureMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 1))

    def forward(self, x):
        global_features = self.avg_pool(x)
        return global_features

class LocalFeatureMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 1))

    def forward(self, x):
        local_features = self.max_pool(x)
        return local_features

def split_layer(input):
    """Splits a layer into global and local features.

    Args:
        input: The input layer.

    Returns:
        A tuple of global and local features.
    """
    C, H, W = input.shape
    global_features = input[:, :, :int(C / 2)]
    local_features = input[:, :, int(C / 2):]

    return global_features, local_features

def extract_global_feature_map(global_features, global_feature_extractor):
    """Extracts global feature map from global features.

    Args:
        global_features: The global features.
        global_feature_extractor: The global feature extractor.

    Returns:
        The global feature map.
    """
    global_feature_map = global_feature_extractor(global_features)
    return global_feature_map

def extract_local_feature_map(local_features, local_feature_extractor):
    """Extracts local feature map from local features.

    Args:
        local_features: The local features.
        local_feature_extractor: The local feature extractor.

    Returns:
        The local feature map.
    """
    local_feature_map = local_feature_extractor(local_features)
    return local_feature_map

def main():
    # Create input layer
    input_tensor = torch.randn(10, 20, 30)

    # Split input layer into global and local features
    global_features, local_features = split_layer(input_tensor)

    # Define global and local feature extractors
    global_feature_extractor = GlobalFeatureMap()
    local_feature_extractor = LocalFeatureMap()

    # Extract global feature map
    global_feature_map = extract_global_feature_map(global_features, global_feature_extractor)

    # Extract local feature map
    local_feature_map = extract_local_feature_map(local_features, local_feature_extractor)

    # Print global and local feature maps
    print("Global feature map:")
    print(global_feature_map)

    print("Local feature map:")
    print(local_feature_map)
    print(input_tensor.shape)
if __name__ == "__main__":
    main()
