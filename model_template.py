from typing import List, Tuple, Dict, Union

import numpy as np


class Model:

    def __init__(
        self,
        name: str,
        category_dict: Dict[int, str],
        trained_on: str,
    ):

        self.name = name
        self.category_dict = category_dict
        self.trained_on = trained_on

    def predict(self, input_data: np.array) -> np.array:
        """
        Run the model on the input data
        """
        pass

    def predict_scale_space(self, scale_space_input):
        """
        Run the model on the input scale space
        """
        pass

    def get_example_images(self):
        """
        Get example images for the model
        """
        pass
    
    def compatible_with(self, model) -> bool:
        """
        Check if the model is compatible with another model
        """
        return self.trained_on == model.trained_on
