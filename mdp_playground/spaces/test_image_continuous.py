import unittest
import numpy as np
from mdp_playground.spaces.image_continuous import ImageContinuous
from gym.spaces import Box

# import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

# import PIL.ImageOps as ImageOps


class TestImageContinuous(unittest.TestCase):
    def test_image_continuous(self):
        # For images in the paper width and height were 100
        # circle_radius was 5
        render = False

        lows = 0.0
        highs = 20.0
        cs2 = Box(
            shape=(2,),
            low=lows,
            high=highs,
        )
        cs4 = Box(
            shape=(4,),
            low=lows,
            high=highs,
        )

        imc = ImageContinuous(
            cs2,
            width=400,
            height=400,
        )
        pos = np.array([5.0, 7.0])
        img1 = Image.fromarray(np.squeeze(imc.generate_image(pos)), "RGB")
        # img1 = ImageOps.invert(img1)
        if render:
            img1.show()
        # img1.save("cont_state_no_target.pdf")

        pos = np.array([10.0, 10.0])
        target = np.array([10, 10])
        imc = ImageContinuous(
            cs2,
            circle_radius=5,
            target_point=target,
            width=100,
            height=100,
        )
        img1 = Image.fromarray(np.squeeze(imc.generate_image(pos)), "RGB")
        if render:
            img1.show()
        img1.save("cont_state_target.pdf")

        # Terminal sub-spaces
        lows = np.array([2.0, 4.0])
        highs = np.array([3.0, 6.0])
        cs2_term1 = Box(
            low=lows,
            high=highs,
        )
        lows = np.array([12.0, 3.0])
        highs = np.array([13.0, 4.0])
        cs2_term2 = Box(
            low=lows,
            high=highs,
        )
        term_spaces = [cs2_term1, cs2_term2]

        target = np.array([10, 10])
        imc = ImageContinuous(
            cs2,
            target_point=target,
            circle_radius=10,
            term_spaces=term_spaces,
            width=400,
            height=400,
        )
        pos = np.array([5.0, 7.0])
        img1 = Image.fromarray(np.squeeze(imc.get_concatenated_image(pos)), "RGB")
        if render:
            img1.show()
        # img1.save("cont_state_target_terminal_states.pdf")

        # Irrelevant features
        target = np.array([10, 10])
        imc = ImageContinuous(
            cs4,
            target_point=target,
            width=400,
            height=400,
        )
        pos = np.array([5.0, 7.0, 10.0, 15.0])
        img1 = Image.fromarray(np.squeeze(imc.get_concatenated_image(pos)), "RGB")
        if render:
            img1.show()
        # print(imc.get_concatenated_image(pos).shape)

        # Random sample and __repr__
        imc = ImageContinuous(
            cs4,
            target_point=target,
            width=400,
            height=400,
        )
        # print(imc)
        img1 = Image.fromarray(np.squeeze(imc.sample()), "RGB")
        if render:
            img1.show()

        # Draw grid
        grid_shape = (5, 5)
        cs2_grid = Box(
            low=0 * np.array(grid_shape).astype(np.float64),
            high=np.array(grid_shape).astype(np.float64),
        )
        pos = np.array([2, 3])
        target = np.array([4, 4])
        imc = ImageContinuous(
            cs2_grid,
            target_point=target,
            circle_radius=10,
            width=400,
            height=400,
            grid_shape=grid_shape,
        )
        img1 = Image.fromarray(np.squeeze(imc.get_concatenated_image(pos)), "RGB")
        if render:
            img1.show()
        # img1.save("grid_target.pdf")

        # Grid with terminal sub-spaces
        lows = np.array([2.0, 4.0])
        highs = np.array([2.0, 4.0])
        cs2_term1 = Box(
            low=lows,
            high=highs,
        )
        lows = np.array([1.0, 1.0])
        highs = np.array([1.0, 1.0])
        cs2_term2 = Box(
            low=lows,
            high=highs,
        )
        term_spaces = [cs2_term1, cs2_term2]

        pos = np.array([2, 3])
        target = np.array([4, 4])
        imc = ImageContinuous(
            cs2_grid,
            circle_radius=10,
            target_point=target,
            term_spaces=term_spaces,
            width=400,
            height=400,
            grid_shape=grid_shape,
        )
        img1 = Image.fromarray(np.squeeze(imc.get_concatenated_image(pos)), "RGB")
        if render:
            img1.show()
        # img1.save("grid_target_terminal_states.pdf")


if __name__ == "__main__":
    unittest.main()
