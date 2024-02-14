import random
from PIL import Image, ImageDraw
import colour
import matplotlib.pyplot
import numpy as np

import deltae


class ImgChromo:
    """
    Defines the individual chromosome (image) to  be evolved.
    This implementation of the chromosome, especially in initialization is mostly adapted from 
    https://medium.com/@sebastian.charmot/genetic-algorithm-for-image-recreation-4ca546454aaa

    """

    def __init__(self, l, w, max_poly_size=6):
        self.max_poly_size = max_poly_size
        self.image = None
        self.l = l
        self.w = w
        self.fitness = float('inf')
        self.img_array = None
        self.create_random_image()

    def rand_color(self):
        # generate random hex values
        return "#" + ''.join([random.choice('0123456789ABCDEF') for color_vals in range(6)])

    def create_random_image(self):
        # number of polygons to add to the image
        iterations = random.randint(3, 50)
        region = (self.l + self.w)//8  # the image is broken up into sections
        img = Image.new("RGBA", (self.l, self.w),
                        self.rand_color())  # (mode, size, color)

        # numebr of points for each polygon
        for i in range(iterations):
            num_points = random.randint(3, 50)

            # the centrepoints on where to create the image
            centrepoint_x = random.randint(0, self.l)
            centrepoint_y = random.randint(0, self.w)

            xy = []
            for j in range(num_points):
                xy.append((random.randint(centrepoint_x - region, centrepoint_x + region),
                           random.randint(centrepoint_y - region, centrepoint_y + region)))

            img1 = ImageDraw.Draw(img)
            img1.polygon(xy, fill=self.rand_color())

        self.image = img
        self.array = self.to_array(img)

    def create_one_color(self):
        self.image = Image.new(mode="RGBA", size=(
            self.l, self.w), color=self.rand_color())

    def create_random_image_array_2(self):
        self.array = np.random.randint(
            low=0, high=255, size=(self.l, self.w, 4))

        self.array = self.array.astype('uint8')

        self.image = Image.fromarray(self.array.astype('uint8'))

    def add_shape(self):
        iterations = random.randint(1, 1)

        region = random.randint(1, (self.l + self.w)//4)

        img = self.image

        for i in range(iterations):
            num_points = random.randint(3, 6)

            region_x = random.randint(0, self.l)
            region_y = random.randint(0, self.w)

            xy = []
            for j in range(num_points):
                xy.append((random.randint(region_x - region, region_x + region),
                           random.randint(region_y - region, region_y + region)))

            img1 = ImageDraw.Draw(img)
            img1.polygon(xy, fill=self.rand_color())

        self.image = img
        self.array = self.to_array(img)

    def to_image(self):
        im = Image.fromarray(self.array)
        im.show()

    def to_array(self, image):
        return np.array(image)

    def compute_fitness(self, target):

        lab1 = np.array(self.array)
        lab2 = np.array(target)
        self.fitness = np.mean(
            colour.difference.delta_E_CIE1976(lab1, lab2))  # delta_E_CIE1976(target, self.array))
        # experimented with delta from various libraries.
        # DeltaE_CIE2000 needed some extra params that I didn't bother to add

    # The other get fitness measures color difference without taking transparency into account takes longer to look right, if it even does look right
    # def get_fitness(self, target):
    #     diff_array = np.subtract(np.array(target), self.array)
    #     self.fitness = np.mean(np.absolute(diff_array))
