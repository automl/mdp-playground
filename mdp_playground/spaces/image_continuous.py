import warnings
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Space
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from PIL.Image import FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM
import os


class ImageContinuous(Box):
    """A space that maps a continuous 1- or 2-D space 1-to-1 to images so that the
    images may be used as representations for corresponding continuous environments.

    Methods
    -------
    get_image_representation(continuous_obs)
        Gets an image representation for a given feature space observation
    """

    def __init__(
        self,
        feature_space,
        term_spaces=None,
        width=100,
        height=100,
        num_channels=3,
        circle_radius=5,
        target_point=None,
        relevant_indices=[0, 1],
        seed=None,
        grid_shape=None,
        dtype=np.uint8,
    ):
        """
        Parameters
        ----------
        feature_space : Gym.spaces.Box
            The feature space to which this class associates images as external
            observations
        term_spaces : list of Gym.spaces.Box
            Sub-spaces of the feature space which are terminal
        width : int
            The width of the image
        height : int
            The height of the image
        num_channels : int
            The number of channels in the image  ###TODO: Support for 1 channel; unify with ImageMultiDiscrete
        circle_radius : int
            The radius of the circle which represents the agent and target point
        target_point : np.array

        relevant_indices : list

        grid_shape : tuple of length 2
            For grid-world environments, the shape of the grid to be drawn
        seed : int
            Seed for this space
        """
        # ##TODO Define a common superclass for this and ImageMultiDiscrete
        self.feature_space = feature_space
        assert (self.feature_space.high != np.inf).any()
        assert (self.feature_space.low != -np.inf).any()
        self.width = width
        self.height = height
        self.num_channels = num_channels
        # Warn if resolution is too low?
        self.circle_radius = circle_radius
        self.target_point = target_point
        self.term_spaces = term_spaces
        self.relevant_indices = relevant_indices
        all_indices = set(range(self.feature_space.shape[0]))
        self.irrelevant_indices = list(all_indices - set(self.relevant_indices))
        if len(self.irrelevant_indices) == 0:
            self.irrelevant_features = False
        else:
            self.irrelevant_features = True
        if grid_shape is not None:
            self.draw_grid = True
            assert type(grid_shape == tuple) and (
                len(grid_shape) == 2 or len(grid_shape) == 4
            )
            # Could also assert that self.width is divisible by grid_shape[0], etc.
            self.grid_shape = grid_shape
        else:
            self.draw_grid = False

        self.goal_colour = (0, 255, 0)
        self.agent_colour = (0, 0, 255)
        self.term_colour = (0, 0, 0)
        self.bg_colour = (208, 208, 208)
        self.line_colour = (255, 255, 255)
        # Alternate scheme
        # self.term_colour = (255, 0, 0)
        # self.bg_colour = (0, 0, 0)

        assert len(feature_space.shape) == 1
        relevant_dims = len(relevant_indices)
        irr_dims = len(self.irrelevant_indices)
        assert relevant_dims <= 2 and irr_dims <= 2, (
            "Image observations are " "supported only " "for 1- or 2-D feature spaces."
        )

        # Shape needs 3rd dimension for Ray Rllib to be compatible IIRC
        super(ImageContinuous, self).__init__(
            shape=(width, height, num_channels), dtype=dtype, low=0, high=255
        )
        super(ImageContinuous, self).seed(seed=seed)

        if self.target_point is not None:
            if self.draw_grid:
                target_point = target_point.astype(float)
                target_point += 0.5
            self.target_point_pixel = self.convert_to_pixel(target_point)

    def generate_image(self, position, relevant=True, epistemic_uncertainty=None):
        """
        Parameters
        ----------
        position : np.array
            A 2-D position in the continuous space
        relevant : bool
            Whether the position is in the relevant sub-space of RLToyEnv or not
        epistemic_uncertainty : np.array
            Assumed to be the std dev. of a Gaussian over the position. If given,
            we draw an uncertainty ellipse over the position.
            

        """
        # Use RGB
        if self.num_channels == 3:
            image_ = Image.new("RGB", (self.width, self.height), color=self.bg_colour)
        elif self.num_channels == 1:
            image_ = Image.new("L", (self.width, self.height), color=self.bg_colour)
        draw = ImageDraw.Draw(image_)

        # Draw in decreasing order of importance:
        # grid lines (in case of grid-based envs), term_spaces, etc. first, so that others are drawn over them
        if self.draw_grid:
            position = position.astype(float)
            position += 0.5
            offset = 0 if relevant else 2
            for i in range(
                1, self.grid_shape[0 + offset] + 1
            ):  # +1 because this is along
                # concatentation dimension when stitching together images below in
                # get_image_representation
                x_ = (
                    i * self.width // self.grid_shape[0 + offset] - 1
                )  # -1 to not go outside
                # image size for the last line drawn
                y_ = self.height
                start_pt = (x_, y_)
                y_ = 0
                end_pt = (x_, y_)
                draw.line([start_pt, end_pt], fill=self.line_colour)

            for j in range(1, self.grid_shape[1 + offset]):
                x_ = self.width
                y_ = j * self.height // self.grid_shape[0 + offset]
                start_pt = (x_, y_)
                x_ = 0
                end_pt = (x_, y_)
                draw.line([start_pt, end_pt], fill=self.line_colour)

        if self.term_spaces is not None and relevant:
            for term_space in self.term_spaces:
                low = self.convert_to_pixel(term_space.low)
                if self.draw_grid:
                    high = self.convert_to_pixel(term_space.high + 1.0)
                else:
                    high = self.convert_to_pixel(term_space.high)

                leftUpPoint = tuple((low))
                rightDownPoint = tuple((high))
                twoPointList = [leftUpPoint, rightDownPoint]
                draw.rectangle(twoPointList, fill=self.term_colour)

        R = self.circle_radius

        # The target point matters only in the relevant sub-space:
        if self.target_point is not None and relevant:
            # print("draw2", self.target_point_pixel)
            leftUpPoint = tuple((self.target_point_pixel - R))
            rightDownPoint = tuple((self.target_point_pixel + R))
            twoPointList = [leftUpPoint, rightDownPoint]
            draw.ellipse(twoPointList, fill=self.goal_colour)

        pos_pixel = self.convert_to_pixel(position)
        # print("draw1", pos_pixel)
        # Draw circle https://stackoverflow.com/a/2980931/11063709
        leftUpPoint = tuple(pos_pixel - R)
        rightDownPoint = tuple(pos_pixel + R)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw.ellipse(twoPointList, fill=self.agent_colour)

        if epistemic_uncertainty is not None:
            epi_unc_pixel = self.convert_to_pixel(epistemic_uncertainty, scale_only=True)
            leftUpPoint = tuple(pos_pixel - R - epi_unc_pixel)
            rightDownPoint = tuple(pos_pixel + R + epi_unc_pixel)
            twoPointList = [leftUpPoint, rightDownPoint]
            draw.ellipse(twoPointList, outline=self.agent_colour)

        # Because numpy is row-major and Image is column major, need to transpose
        # ret_arr = np.array(image_).T # For 2-D
        ret_arr = np.transpose(np.array(image_), axes=(1, 0, 2))

        return ret_arr

    def get_image_representation(self, obs):
        """Gets the "stitched together" image made from images corresponding to
        each continuous sub-space within the continuous space, concatenated
        along the X-axis.

        obs can be a single 2-D observation vector or a 2-D tensor / matrix with 
        observations along the 2nd axis. If it is such a tensor, we take the mean 
        and std dev. of the tensor and generate an image with an uncertainty
        over the state.
        """

        # Check if obs is a 2-D tensor of observations, obtained possibly from an ensemble,
        # so as to estimate some level of epistemic uncertainty from them:
        if len(obs.shape) == 2:
            epi_unc = True
            mean = np.mean(obs, axis=0)
            std_dev = np.std(obs, axis=0)
            obs = mean
        else:
            epi_unc = False

        concatenated_image = []
        # For relevant/irrelevant sub-spaces:
        concatenated_image.append(self.generate_image(obs[self.relevant_indices], 
                                                      epistemic_uncertainty=std_dev[self.relevant_indices] 
                                                      if epi_unc else None))
        if self.irrelevant_features:
            irr_image = self.generate_image(
                obs[self.irrelevant_indices], relevant=False, epistemic_uncertainty=std_dev[self.irrelevant_indices]
                if epi_unc else None
            )
            concatenated_image.append(irr_image)

        concatenated_image = np.concatenate(tuple(concatenated_image), axis=0)

        return np.atleast_3d(concatenated_image)  # because Ray expects an
        # image to have >=3 dims

    def convert_to_pixel(self, vector, scale_only=False):
        """
        Converts a continuous vector from the feature space of the object 
        to an integer pixel position in the image representation space by default.
        If scale_only is True, we return the vector scaled by the ratio of
        image size to feature space size.

        Parameters
        ----------
        vector : np.array
            A 2-D vector in the feature space that can represent a position
            or anything else in the feature space, e.g., std. dev. of a
            Gaussian over the position.
        scale_only : bool
            If True, we only scale

        """
        # It's implicit that both relevant and irrelevant sub-spaces have the
        # same max and min here:
        max = self.feature_space.high[self.relevant_indices]
        min = self.feature_space.low[self.relevant_indices]
        if scale_only:
            # By default this is used for scaling uncertainty (std dev), so we heuristically multiply by 3
            # to make it look good. 1 std dev, in some cases, led to a 0 pixel std dev after converting to int.
            pixel_vec = 3 * (vector) / (max - min)
        else:
            pixel_vec = (vector - min) / (max - min)
        pixel_vec = (pixel_vec * self.shape[:2]).astype(int)  # self.shape is (100, 100, 3) by default

        return pixel_vec

    def sample(self):

        sampled = self.feature_space.sample()
        return self.get_image_representation(sampled)

    def __repr__(self):
        return (
            "{} with continuous underlying space of shape: {} and "
            "images of resolution: {} and dtype: {}".format(
                self.__class__, self.feature_space.shape, self.shape, self.dtype
            )
        )

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        if x.shape == (
            self.width,
            self.height,
            self.num_channels,
        ):  # TODO compare each pixel for all possible image observations? Hard to implement.
            return True

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        raise NotImplementedError

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError
