import unittest
import numpy as np
from mdp_playground.spaces.image_continuous import ImageContinuous
from gym.spaces import Box
# import PIL.ImageDraw as ImageDraw
import PIL.Image as Image


class TestImageContinuous(unittest.TestCase):

    def test_image_continuous(self):
        lows = 0.0
        highs = 20.0
        cs2 = Box(shape=(2,), low=lows, high=highs,)
        cs4 = Box(shape=(4,), low=lows, high=highs,)

        imc = ImageContinuous(cs2, width=400, height=400,)
        pos = np.array([5.0, 7.0])
        img1 = Image.fromarray(np.squeeze(imc.generate_image(pos)), 'RGB')
        img1.show()

        target = np.array([10, 10])
        imc = ImageContinuous(cs2, target_point=target, width=400, height=400,)
        img1 = Image.fromarray(np.squeeze(imc.generate_image(pos)), 'RGB')
        img1.show()

        # Terminal sub-spaces
        lows = np.array([2., 4.])
        highs = np.array([3., 6.])
        cs2_term1 = Box(low=lows, high=highs,)
        lows = np.array([12., 3.])
        highs = np.array([13., 4.])
        cs2_term2 = Box(low=lows, high=highs,)
        term_spaces = [cs2_term1, cs2_term2]

        target = np.array([10, 10])
        imc = ImageContinuous(cs2, target_point=target, term_spaces=term_spaces,\
                        width=400, height=400,)
        pos = np.array([5.0, 7.0])
        img1 = Image.fromarray(np.squeeze(imc.get_concatenated_image(pos)), 'RGB')
        img1.show()


        # Irrelevant features
        target = np.array([10, 10])
        imc = ImageContinuous(cs4, target_point=target, width=400, height=400,)
        pos = np.array([5.0, 7.0, 10.0, 15.0])
        img1 = Image.fromarray(np.squeeze(imc.get_concatenated_image(pos)), 'RGB')
        img1.show()
        # print(imc.get_concatenated_image(pos).shape)

        # Random sample and __repr__
        imc = ImageContinuous(cs4, target_point=target, width=400, height=400,)
        print(imc)
        img1 = Image.fromarray(np.squeeze(imc.sample()), 'RGB')
        img1.show()




if __name__ == '__main__':
    unittest.main()
