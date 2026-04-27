"""Utility module for loading a pre-trained CHGNet model from matgl."""

from matgl.models import CHGNet


def load_pretrained_chgnet(model_name: str = "CHGNet-PES") -> CHGNet:
    """Load a pre-trained CHGNet model from matgl.

    Args:
        model_name (str): The name of the pre-trained model to load.
            Defaults to "CHGNet-PES".

    Returns:
        CHGNet: The loaded model instance.
    """
    return CHGNet.from_pretrained(model_name)


if __name__ == "__main__":
    model = load_pretrained_chgnet()
    print(f"Loaded CHGNet model: {model.__class__.__name__}")
