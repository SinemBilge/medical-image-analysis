import torch

import numpy as np

from model_template import Model
from UNetJonathan import UNet
import os


STANDARD_DICT = {
    0: {
        "name": "background",
        "abbreviation": "bg",
        "color": (0.0, 0.0, 0.0, 0.0),
    },
    1: {
        "name": "Left Ventricle Endocardium",
        "abbreviation": "LVEn",
        "color": (1.0, 0.0, 0.0, 0.4),
    },
    2: {
        "name": "Left Ventricle Epicardium",
        "abbreviation": "LVEp",
        "color": (0.0, 0.0, 1.0, 0.4),
    },
    3: {
        "name": "Right Ventricle Endocardium",
        "abbreviation": "RVEn",
        "color": (1.0, 1.0, 0.0, 0.4),
    },
}


def modify_state_dict(state_dict):
    """
    Modify the state dict to be compatible with the model
    """

    print("Modifying state dict")

    unexpected = [
        "model.1.submodule.conv.unit0.conv.weight",
        "model.1.submodule.conv.unit0.conv.bias",
        "model.1.submodule.conv.unit0.adn.A.weight",
        "model.1.submodule.conv.unit1.conv.weight",
        "model.1.submodule.conv.unit1.conv.bias",
        "model.1.submodule.conv.unit1.adn.A.weight",
        "model.1.submodule.conv.unit2.conv.weight",
        "model.1.submodule.conv.unit2.conv.bias",
        "model.1.submodule.conv.unit2.adn.A.weight",
        "model.1.submodule.conv.unit3.conv.weight",
        "model.1.submodule.conv.unit3.conv.bias",
        "model.1.submodule.conv.unit3.adn.A.weight",
        "model.1.submodule.1.submodule.conv.unit0.conv.weight",
        "model.1.submodule.1.submodule.conv.unit0.conv.bias",
        "model.1.submodule.1.submodule.conv.unit0.adn.A.weight",
        "model.1.submodule.1.submodule.conv.unit1.conv.weight",
        "model.1.submodule.1.submodule.conv.unit1.conv.bias",
        "model.1.submodule.1.submodule.conv.unit1.adn.A.weight",
        "model.1.submodule.1.submodule.conv.unit2.conv.weight",
        "model.1.submodule.1.submodule.conv.unit2.conv.bias",
        "model.1.submodule.1.submodule.conv.unit2.adn.A.weight",
        "model.1.submodule.1.submodule.conv.unit3.conv.weight",
        "model.1.submodule.1.submodule.conv.unit3.conv.bias",
        "model.1.submodule.1.submodule.conv.unit3.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.residual.weight",
        "model.1.submodule.1.submodule.1.submodule.0.residual.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit0.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit0.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit0.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit1.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit1.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit1.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit2.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit2.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit2.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit3.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit3.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit3.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit0.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit0.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit0.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit1.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit1.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit1.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit2.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit2.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit2.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit3.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit3.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit3.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit4.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit4.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit4.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit5.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit5.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit5.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit6.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit6.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit6.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit7.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit7.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit7.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.residual.weight",
        "model.1.submodule.1.submodule.1.submodule.residual.bias",
    ]

    missing = [
        "model.1.submodule.0.conv.unit0.conv.weight",
        "model.1.submodule.0.conv.unit0.conv.bias",
        "model.1.submodule.0.conv.unit0.adn.A.weight",
        "model.1.submodule.0.conv.unit1.conv.weight",
        "model.1.submodule.0.conv.unit1.conv.bias",
        "model.1.submodule.0.conv.unit1.adn.A.weight",
        "model.1.submodule.0.conv.unit2.conv.weight",
        "model.1.submodule.0.conv.unit2.conv.bias",
        "model.1.submodule.0.conv.unit2.adn.A.weight",
        "model.1.submodule.0.conv.unit3.conv.weight",
        "model.1.submodule.0.conv.unit3.conv.bias",
        "model.1.submodule.0.conv.unit3.adn.A.weight",
        "model.1.submodule.1.submodule.0.conv.unit0.conv.weight",
        "model.1.submodule.1.submodule.0.conv.unit0.conv.bias",
        "model.1.submodule.1.submodule.0.conv.unit0.adn.A.weight",
        "model.1.submodule.1.submodule.0.conv.unit1.conv.weight",
        "model.1.submodule.1.submodule.0.conv.unit1.conv.bias",
        "model.1.submodule.1.submodule.0.conv.unit1.adn.A.weight",
        "model.1.submodule.1.submodule.0.conv.unit2.conv.weight",
        "model.1.submodule.1.submodule.0.conv.unit2.conv.bias",
        "model.1.submodule.1.submodule.0.conv.unit2.adn.A.weight",
        "model.1.submodule.1.submodule.0.conv.unit3.conv.weight",
        "model.1.submodule.1.submodule.0.conv.unit3.conv.bias",
        "model.1.submodule.1.submodule.0.conv.unit3.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.residual.weight",
        "model.1.submodule.1.submodule.1.submodule.residual.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit0.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit0.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit0.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit1.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit1.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit1.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit2.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit2.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit2.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit3.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.conv.unit3.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.conv.unit3.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit0.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit0.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit0.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit1.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit1.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit1.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit2.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit2.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit2.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit3.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit3.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit3.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit4.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit4.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit4.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit5.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit5.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit5.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit6.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit6.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit6.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit7.conv.weight",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit7.conv.bias",
        "model.1.submodule.1.submodule.1.submodule.0.conv.unit7.adn.A.weight",
        "model.1.submodule.1.submodule.1.submodule.0.residual.weight",
        "model.1.submodule.1.submodule.1.submodule.0.residual.bias",
    ]

    translate_keys_dict = dict(zip(unexpected, missing))

    # Fix the state dict keys by removing the extra '0' from the keys
    fixed_state_dict = {}
    for key in state_dict["model_state_dict"]:
        # Replace ".0." with ".", which should fix the naming mismatch
        new_key = key
        if key in translate_keys_dict:
            new_key = translate_keys_dict[key]
        fixed_state_dict[new_key] = state_dict["model_state_dict"][key]

    return fixed_state_dict


class MonaiUNet(Model):
    def __init__(self, name, state_dict_path, n_filters_init, depth, num_res_units):

        super().__init__(name, STANDARD_DICT, "Heart500")

        ## fixed params
        self.in_channels = 1
        self.out_channels = 4

        self.n_filters_init = n_filters_init
        self.depth = depth
        self.num_res_units = num_res_units

        ## derived params
        self.channels = [n_filters_init * 2**i for i in range(depth)]
        self.strides = [2] * (depth - 1)

        self.model = UNet(
            spatial_dims=2,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=self.channels,
            strides=self.strides,
            num_res_units=num_res_units,
        )

        # state_dict = torch.load(state_dict_path, weights_only=False)

        # try:
        #     self.model.load_state_dict(state_dict["model_state_dict"])
        # except RuntimeError:
        #     self.model.load_state_dict(modify_state_dict(state_dict))
        # Force CPU loading
        state_dict = torch.load(state_dict_path, map_location='cpu')

        try:
        # Try normal checkpoint format
            self.model.load_state_dict(state_dict["model_state_dict"])
        except RuntimeError:
            self.model.load_state_dict(modify_state_dict(state_dict))

        self.model.eval()

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run the model on the input data
        """

        assert (
            type(input_data) == np.ndarray
        ), f"Expected np.ndarray but got {type(input_data)}"
        assert (
            len(input_data.shape) == 3
        ), f"(width, height, slide_nr) but got {input_data.shape} comment: maybe an error: i think the slide_nr shoudld be first"

        width = input_data.shape[1]
        height = input_data.shape[2]

        assert (
            width <= 256 and height <= 256
        ), f"Expected width and height to be <= 256, but got {width = } and {height = }"

        slide_number = input_data.shape[0]

        # Normalize the input data between 0 and 1
        input_data = (input_data - np.min(input_data)) / (
            np.max(input_data) - np.min(input_data)
        )

        x_in = torch.from_numpy(input_data).unsqueeze(0).float()
        x_in = x_in.view(slide_number, 1, width, height)

        desired_size = 256

        # Calculate the padding needed for each dimension
        pad_width = (desired_size - width) // 2
        pad_height = (desired_size - height) // 2

        # If the size difference is odd, add an extra pixel of padding on one side
        padding = (
            pad_height,
            desired_size - height - pad_height,
            pad_width,
            desired_size - width - pad_width,
        )

        # Apply padding
        x_in_padded = torch.nn.functional.pad(x_in, padding, mode="constant", value=0)

        x_out_unordered = self.model(x_in_padded).detach().numpy()

        x_out_unpadded = x_out_unordered[
            :, :, pad_width : pad_width + width, pad_height : pad_height + height
        ]

        return x_out_unpadded

    def predict_scale_space(self, scale_space_input: np.ndarray) -> np.ndarray:
        """
        Run the model on the input scale space
        """

        assert (
            type(scale_space_input) == np.ndarray
        ), f"Expected np.ndarray but got {type(scale_space_input)}"
        assert (
            len(scale_space_input.shape) == 4
        ), f"(n, width, height, slide_nr) but got {scale_space_input.shape}"

        if (
            len(scale_space_input.shape) == 3
            or scale_space_input.shape[2] < 150
            or scale_space_input.shape[3] < 150
        ):
            raise Exception(
                f"Invalid shape for scale_space_input: {scale_space_input.shape}, should be (n, width, height) (one image per scale) or (n, x, width, height) (x images per scale)"
            )

        width = scale_space_input.shape[2]
        height = scale_space_input.shape[3]

        assert (
            width <= 256 and height <= 256
        ), f"Expected width and height to be <= 256, but got {width = } and {height = }"

        slide_number = scale_space_input.shape[1]

        x_out = np.zeros((len(scale_space_input), slide_number, 4, width, height))

        for i, scale_space_data in enumerate(scale_space_input):

            x_out[i] = self.predict(scale_space_data)

        return x_out


class UNetHeart_8_4_4(MonaiUNet):
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "models")
        model_path = os.path.join(model_dir, "heart_monai-8-4-4_all_0_best.pt")

        super().__init__(
            "UNetHeart_8_4_4",
            model_path,
            n_filters_init=8,
            depth=4,
            num_res_units=4,
        )


class UNetHeart_16_4_8(MonaiUNet):

    def __init__(self):
        super().__init__(
            "UNetHeart_16_4_8",
            "models/heart_monai-16-4-8_all_0_best.pt",
            n_filters_init=16,
            depth=4,
            num_res_units=8,
        )
