import warnings
import numpy as np
import gym
from gym.spaces import Box, Discrete, MultiDiscrete, Space
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from PIL.Image import FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM
import os


class ImageMultiDiscrete(Box):
    """A space that maps a (multi-)discrete space 1-to-1 to images so that the images may be used as representations for corresponding (multi-)discrete states. A MultiDiscrete space will have multiple dimensions. For each of these dimensions, there is a size that represents the number of categorical states that correspond to that dimension. For size = n, each of these categorical states is numbered from 0 to n-1. For each categorical state numbered n, we associate a polygon with n + 3 sides. This polygon is present in the image associated with this dimension. The images generated for all the dimensions are concatenated together by placing them side by side in the order of the dimensions in Space. Any of the transforms - rotate, flip, scale, shift - can be associated with an object of this class, to apply at random to polygons in the images whenever they are generated.

    Methods
    -------
    get_concatenated_image(multi_discrete_state)
        Gets an image representation for a given multi_discrete_state
    """

    def __init__(
        self,
        state_space_sizes,
        width=100,
        height=100,
        circle_radius=20,
        transforms="rotate,flip,scale,shift",
        sh_quant=1,
        scale_range=(0.5, 1.5),
        ro_quant=1,
        seed=None,
        use_custom_images=None,
        cust_path=None,
        dtype=np.uint8,
    ):  # , polygon_sides=4
        """
        Parameters
        ----------
        state_space_sizes : list
            The underlying (multi-)discrete state space sizes to which this class associates images as external observations
        width : int
            The width of the image
        height : int
            The height of the image
        circle_radius : int
            The radius of the circle in which the associated polygons are inscribed
        transforms : str
            Comma separated string specifiying which transforms are applied to images. (Commas are not actually needed, but are recommended for readability, since it's only the presence of the string representing the corresponding transform that is checked for.)
        sh_quant : int
            An int to quantise the shift transforms.
        scale_range: tuple of floats with length = 2
            A tuple of real numbers to specify (min_scaling, max_scaling) for the scale transform.
        ro_quant : int
            An int to quantise the rotation transforms.
        seed : int
            seed for randomly applied transformations and NOT for the underlying state space
        use_custom_images : str or None
            If None, then default setting of no custom textures or images. If this value is "textures" or "images", then all images in the cust_path directories are loaded in alphabetical order and correspond 1-to-1 with discrete states which are in numeric order. If this value is "textures", the textures are applied to the polygons that would have been generated for the default setting (of no custom textures or images). If this value is "images", then the custom images are drawn in a square (with side length = circle_radius * sqrt(2)) in the centre of the polygon when no transforms are applied. When the underlying state space sizes are multi-discrete, the 1-to-1 state number to image mapping is the same for all discrete sub-spaces.
        cust_path : str or None
            The directory containing the custom images to be loaded
        """
        self.width = width
        self.height = height
        # Warn if resolution is too low?
        self.circle_radius = circle_radius
        self.transforms = transforms
        self.sh_quant = sh_quant
        self.ro_quant = ro_quant
        self.scale_range = scale_range

        # self.state_space = state_space
        self.use_custom_images = use_custom_images

        # if isinstance(state_space, Discrete):
        #     state_space_sizes = [state_space.n] # can be an int to map images to discrete spaces or it can be a list to map images (each image = multiple images, one for each discrete dimension, concatenated along the X-axis) to multi-discrete spaces
        # elif isinstance(state_space, MultiDiscrete):
        #     state_space_sizes = list(state_space.nvec)
        # else:
        #     raise TypeError('ImageMultiDiscrete can only hold a Discrete or MultiDiscrete object from Gym. Provided object was of type: ' + str(type(state_space)))

        self.state_space_sizes = state_space_sizes

        if use_custom_images is not None:  # TODO test for textures, custom images
            cust_imgs = []
            # cust_arrs = []
            for img_file in sorted(os.listdir(cust_path)):
                img_ = Image.open(cust_path + "/" + img_file)
                cust_imgs.append(img_)
                # The following code might be used to resize the texture to a canonical size here already or to tile the texture.
                # sq_width = circle_radius * 2 # np.sqrt(2)
                # self.sq_width = sq_width
                # img_ = img_.resize((sq_width, sq_width))
                # cust_imgs.append(img_)
                # img_arr = np.array(img_)
                # cust_arrs.append(img_arr)
            assert len(cust_imgs) > max(
                state_space_sizes
            ), "cust_path should be a directory with at least as many as the larget Discrete sub-space in the MultiDiscrete space."
            # "The cust_path should be a directory with only texture images, at least as many as the larget Discrete sub-space in the MultiDiscrete space."
            self.cust_imgs = cust_imgs

        # self.shape = (width, height, 1)
        super(ImageMultiDiscrete, self).__init__(
            shape=(width, height, 1), dtype=dtype, low=0, high=255
        )  #
        super(ImageMultiDiscrete, self).seed(seed=seed)  #

    # def seed(self, seed=None):
    #     pass

    # def generate_images(self, state_space_size): #, polygon_sides
    #     states_ = np.ndarray(shape=(state_space_size, self.width, self.height), dtype=np.uint8) # can't use 1 for 4th dim of array as, for L grayscale colour space, converting to np.array() gives only 2-D which throws broadcast error when assigning to this array below
    #     for curr_state in range(state_space_size):
    #         states_[curr_state] = self.generate_image(curr_state) # , curr_state + 2 - At least 2 sides for a polygon (including a line as a polygon here), so + 2.
    #
    #     return states_

    def generate_image(self, discrete_state):  # , state_space_size, polygon_sides
        polygon_sides = discrete_state + 3
        sh_quant = self.sh_quant
        ro_quant = self.ro_quant
        scale_range = self.scale_range

        if self.use_custom_images is not None:  # textures / custom images
            image_ = Image.new(
                "RGB", (self.width, self.height)
            )  # Use RGB for textures / custom images
        else:
            image_ = Image.new(
                "L", (self.width, self.height)
            )  # Use L for black and white 8-bit pixels instead of RGB in case not using custom images
        draw = ImageDraw.Draw(image_)

        R = self.circle_radius
        shift_w = int(self.width / 2)
        shift_h = int(self.height / 2)

        if "scale" in self.transforms:
            # max_R = 0.6 * min(self.width, self.height) / 2 # Not sure whether to make this depend on provided R as well
            # min_R = 0.1 * min(self.width, self.height) / 2 # /2 because it's R, 0.6
            # and 0.1 to allow some wiggle for shift below and not make too small
            max_R = scale_range[1] * R
            if int(max_R) > min(self.width, self.height) / 2:
                warnings.warn(
                    "Maximum possible size of polygon might be too big for the given resolution. It's set to: "
                    + str(max_R)
                )
            max_R = np.log(max_R)
            min_R = scale_range[0] * R
            if int(min_R) < 3:
                warnings.warn(
                    "Minimum possible size of polygon might be too small and lead too much noise in image. It's set to: "
                    + str(min_R)
                )
            min_R = np.log(min_R)
            log_sample = min_R + self.np_random.random() * (max_R - min_R)
            sample_ = np.exp(log_sample)
            R = int(sample_)
            # print("R", min_R, max_R)

        if "shift" in self.transforms:
            max_shift_w = self.width / 2 - R
            max_shift_h = self.height / 2 - R
            add_shift_w = self.np_random.randint(-max_shift_w + 1, max_shift_w)
            add_shift_h = self.np_random.randint(-max_shift_h + 1, max_shift_h)
            add_shift_w = (add_shift_w // sh_quant) * sh_quant
            add_shift_h = (add_shift_h // sh_quant) * sh_quant
            # print("shift_w, shift_h", add_shift_w, add_shift_h)
            shift_w += add_shift_w
            shift_h += add_shift_h

        if self.use_custom_images == "images":
            pass
        else:
            # Trace polygon
            points_ = []
            for i in range(polygon_sides):
                angle = (2 * np.pi / polygon_sides) * i
                point = (
                    int(shift_w + R * np.cos(angle)),
                    int(shift_h + R * np.sin(angle)),
                )
                points_.append(point)

        # Render polygon if using default or textures, else use custom image
        if self.use_custom_images == "textures":
            draw.polygon(points_, fill=(255, 255, 255))
            img_arr_ = np.array(image_)
            tex_img = self.cust_imgs[discrete_state]
            tex_img = tex_img.resize((R * 2, R * 2))
            tex_arr = np.array(tex_img)
            top_left = (
                shift_h - tex_arr.shape[0] // 2,
                shift_w - tex_arr.shape[1] // 2,
            )
            bottom_right = (
                shift_h + tex_arr.shape[0] // 2,
                shift_w + tex_arr.shape[1] // 2,
            )
            img_arr_[
                top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]
            ] //= 255
            # //255 to make white pixels be (1, 1, 1) - so when multiplied it's a mask; shift_h and shift_w interchanged as numpy is row major
            img_arr_[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]] *= tex_arr
            # texed_img =
            image_ = Image.fromarray(img_arr_, "RGB")
        elif self.use_custom_images == "images":
            img_arr_ = np.array(image_)
            tex_img = self.cust_imgs[discrete_state]
            sq_width = int(
                R * np.sqrt(2)
            )  # For textures it is not square root because polygons like a pentagon would go outside the sqrt(2) region.
            if (
                sq_width % 2 == 1
            ):  # If sq_width is not even, it causes errors with the //2 below.
                sq_width += 1
            tex_img = tex_img.resize((sq_width, sq_width))
            tex_arr = np.array(tex_img)
            top_left = (
                shift_h - tex_arr.shape[0] // 2,
                shift_w - tex_arr.shape[1] // 2,
            )
            bottom_right = (
                shift_h + tex_arr.shape[0] // 2,
                shift_w + tex_arr.shape[1] // 2,
            )
            img_arr_[
                top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]
            ] = tex_arr
            image_ = Image.fromarray(img_arr_, "RGB")
        else:
            draw.polygon(points_, fill=(255))

        if (
            "rotate" in self.transforms
        ):  # TODO rotation can lead to image going out of bounds.
            # rotation_ = (360 / polygon_sides) * (discrete_state / state_space_size) # Need to divide by polygon_sides because
            rotation = self.np_random.randint(360)
            rotation = (rotation // ro_quant) * ro_quant
            # print("rotation", rotation)
            image_ = image_.rotate(rotation)
            # image_.rotate(

        if "flip" in self.transforms:
            if self.np_random.randint(2) == 0:  # Only flip half the times
                if self.np_random.randint(2) == 0:
                    image_ = image_.transpose(FLIP_LEFT_RIGHT)
                else:
                    image_ = image_.transpose(FLIP_TOP_BOTTOM)

        # Because numpy is row-major and Image is column major, need to transpose
        if self.use_custom_images is None:
            ret_arr = np.array(image_).T
        else:
            ret_arr = np.transpose(np.array(image_), axes=(1, 0, 2))

        return ret_arr

    def get_concatenated_image(
        self,
        multi_discrete_state,
    ):
        """Gets the "stitched together" image made from images corresponding to each discrete sub-space within the multidiscrete space, concatenated along the X-axis"""
        if isinstance(multi_discrete_state, int):
            multi_discrete_state = [multi_discrete_state]
        concatenated_image = []
        for i in range(len(self.state_space_sizes)):  # For each Discrete sub-space
            concatenated_image.append(self.generate_image(multi_discrete_state[i]))
        # for i in range(len(self.disjoint_states)):
        #     concatenated_image.append(self.disjoint_states[i][multi_discrete_state[i]])
        concatenated_image = np.concatenate(tuple(concatenated_image), axis=0)

        return np.atleast_3d(
            concatenated_image
        )  # because Ray expects an image to have >=3 dims

    # def get_multi_discrete_state(self,

    def sample(self):
        sss = np.array(self.state_space_sizes)
        sampled = (self.np_random.random_sample(sss.shape) * sss).astype(
            self.dtype
        )  # Based on Gym's MultiDiscrete sampling
        # if type(sampled) == int:
        #     sampled = [sampled]
        sampled = list(sampled)

        return self.get_concatenated_image(sampled)

    def __repr__(self):
        return (
            "{} with multi-discrete space of shape: {} and "
            "images of resolution: {} and dtype: {}".format(
                self.__class__, self.state_space_sizes, self.shape, self.dtype
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
            1,
        ):  # TODO compare each pixel for all possible images?
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
