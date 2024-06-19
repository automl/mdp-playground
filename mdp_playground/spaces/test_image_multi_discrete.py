import unittest
import numpy as np
from mdp_playground.spaces.image_multi_discrete import ImageMultiDiscrete
from gymnasium.spaces import Discrete, MultiDiscrete

# import gymnasium as gym
# from gymnasium.spaces import MultiDiscrete
# # from .space import Space
# import PIL.ImageDraw as ImageDraw
# import PIL.Image as Image


class TestImageMultiDiscrete(unittest.TestCase):
    def test_image_multi_discrete(self):
        render = False

        ds4 = Discrete(4)
        ds4 = [ds4.n]
        print(ds4)
        imd = ImageMultiDiscrete(ds4, transforms="shift")
        print("imd.dtype, shape", imd.dtype, imd.shape)
        print("imd", imd)
        from PIL import Image

        # img1 = Image.fromarray(imd.disjoint_states[0][1], 'L')
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(3)), "L")
        if render:
            img1.show()

        if render:
            imd = ImageMultiDiscrete(
                ds4,
                transforms="scale,shift,rotate,flip",
                use_custom_images="textures",
                cust_path="/home/rajanr/textures",
            )
            img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(2)), "RGB")
        if render:
            img1.show()

        if render:
            imd = ImageMultiDiscrete(
                ds4,
                transforms="scale,shift,rotate,flip",
                use_custom_images="images",
                cust_path="/home/rajanr/textures",
            )
            img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(1)), "RGB")
        if render:
            img1.show()

        imd = ImageMultiDiscrete(ds4, transforms="scale")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(3)), "L")
        if render:
            img1.show()

        imd = ImageMultiDiscrete(ds4, transforms="rotate")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(3)), "L")
        if render:
            img1.show()

        imd = ImageMultiDiscrete(ds4, transforms="flip")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(3)), "L")
        if render:
            img1.show()

        imd = ImageMultiDiscrete(ds4, transforms="shift")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(2)), "L")
        if render:
            img1.show()
        imd = ImageMultiDiscrete(ds4, transforms="scale")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(2)), "L")
        if render:
            img1.show()
        imd = ImageMultiDiscrete(ds4, transforms="rotate")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(2)), "L")
        if render:
            img1.show()
        imd = ImageMultiDiscrete(ds4, transforms="flip")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(2)), "L")
        if render:
            img1.show()

        imd = ImageMultiDiscrete(ds4, transforms="shift")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(1)), "L")
        if render:
            img1.show()
        imd = ImageMultiDiscrete(ds4, transforms="scale")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(1)), "L")
        if render:
            img1.show()
        imd = ImageMultiDiscrete(ds4, transforms="rotate")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(1)), "L")
        if render:
            img1.show()
        imd = ImageMultiDiscrete(ds4, transforms="flip")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(1)), "L")
        if render:
            img1.show()

        imd = ImageMultiDiscrete(ds4, transforms="shift")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(0)), "L")
        if render:
            img1.show()
        imd = ImageMultiDiscrete(ds4, transforms="scale")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(0)), "L")
        if render:
            img1.show()
        imd = ImageMultiDiscrete(ds4, transforms="rotate")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(0)), "L")
        if render:
            img1.show()
        imd = ImageMultiDiscrete(ds4, transforms="flip")
        img1 = Image.fromarray(np.squeeze(imd.get_concatenated_image(0)), "L")
        if render:
            img1.show()

        # imd = ImageMultiDiscrete(ds4, 100, 100)
        # from PIL import Image
        # img1 = Image.fromarray(imd.disjoint_states[0][1], 'L')


if __name__ == "__main__":
    unittest.main()
