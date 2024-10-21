import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import random
import numpy as np


class ImageAugment(object):
    ImageAugment is a class for augmenting training data using the imgaug library.

    Attributes:
        key (int): A key used for random decisions in augmentation.
        choice (int): A choice parameter for selecting augmentation functions.
        rotate (int): Random rotation angle between -15 and 15 degrees.
        scale_x (float): Random scaling factor for the x-axis between 0.8 and 1.0.
        scale_y (float): Random scaling factor for the y-axis between 0.8 and 1.0.
        translate_x (float): Random translation percentage for the x-axis between -0.1 and 0.1.
        translate_y (float): Random translation percentage for the y-axis between -0.1 and 0.1.
        brightness (int): Random brightness adjustment between -10 and 10.
        linear_contrast (float): Random linear contrast adjustment between 0.5 and 2.0.
        alpha (float): Random alpha value for sharpening between 0 and 1.0.
        lightness (float): Random lightness value for sharpening between 0.75 and 1.5.
        Gaussian (float): Random Gaussian noise value between 0.0 and 12.75.
        Gaussian_blur (float): Random Gaussian blur sigma between 0 and 3.0.

    Methods:
        aug(image, sequence):
            Augments a given image using a specified sequence of augmentation functions.
            
            Parameters:
                image (numpy.ndarray): The image to be augmented, with size (H, W, C).
                sequence (iaa.Sequential): A collection of augmentation functions.
            
            Returns:
                numpy.ndarray: The augmented image.

        rd(hehe):
            Generates a random integer between 0 and a specified upper limit.
            
            Parameters:
                hehe (int): The upper limit for the random integer.
            
            Returns:
                int: A random integer between 0 and hehe.

        aug_sequence():
            Creates a sequence of augmentation functions.
            
            Returns:
                iaa.Sequential: A sequence of augmentation functions in random order.

        aug_function():
            Defines a list of augmentation functions based on random decisions.
            
            Returns:
                list: A list of augmentation functions.
    """
    class for augment the training data using imgaug
    """
    def __init__(self):
        self.key = 0
        self.choice = 1
        self.rotate = np.random.randint(-15, 15)
        self.scale_x = random.uniform(0.8, 1.0)
        self.scale_y = random.uniform(0.8, 1.0)
        self.translate_x = random.uniform(-0.1, 0.1)
        self.translate_y = random.uniform(-0.1, 0.1)
        self.brightness = np.random.randint(-10, 10)
        self.linear_contrast = random.uniform(0.5, 2.0)
        self.alpha = random.uniform(0, 1.0)
        self.lightness = random.uniform(0.75, 1.5)
        self.Gaussian = random.uniform(0.0, 0.05*255)
        self.Gaussian_blur = random.uniform(0, 3.0)

    def aug(self, image, sequence):
        """
        :param image: need size (H, W, C) one image once
        :param label: need size same as image or (H, W)(later translate to size(1, H, W, C))
        :param sequence: collection of augment function
        :return:
        """
        image_aug = sequence(image=image)
        return image_aug

    def rd(self, hehe):
        seed = np.random.randint(0, hehe)
        return seed

    def aug_sequence(self):
        sequence = self.aug_function()
        seq = iaa.Sequential(sequence, random_order=True)
        return seq

    def aug_function(self):
        sequence = []
        if self.rd(2) == self.key:
            sequence.append(iaa.Fliplr(1.0))  # 50% horizontally flip all batch images
        if self.rd(2) == self.key:
            sequence.append(iaa.Flipud(1.0))  # 50% vertically flip all batch images
        if self.rd(2) == self.key:
            sequence.append(iaa.Affine(
                scale={"x": self.scale_x, "y": self.scale_y},  # scale images to 80-100% of their size
                translate_percent={"x": self.translate_x, "y": self.translate_y},  # translate by -10 to +10 percent (per axis)
                rotate=(self.rotate),  # rotate by -15 to +15 degrees
            ))
        if self.rd(2) == self.key:
            sequence.extend(iaa.SomeOf((1, self.choice),
                                       [
                                           iaa.OneOf([
                                               iaa.GaussianBlur(self.Gaussian_blur),  # blur images with a sigma between 0 and 3.0
                                               # iaa.AverageBlur(k=(2, 7)),  # blur images using local means with kernel size 2-7
                                               # iaa.MedianBlur(k=(3, 11))  # blur images using local medians with kernel size 3-11
                                           ]),
                                           # iaa.Sharpen(alpha=self.alpha, lightness=self.lightness),  # sharpen images
                                           # iaa.LinearContrast(self.linear_contrast, per_channel=0.5),  # improve or worse the contrast
                                           # iaa.Add(self.brightness, per_channel=0.5),  # change brightness
                                           # iaa.AdditiveGaussianNoise(loc=0, scale=0.1, per_channel=0.5)  # add gaussian n
                                       ]))
        return sequence


def show_aug(image):
    plt.figure(figsize=(10, 10), facecolor="#FFFFFF")
    for i in range(1, len(image)+1):
        plt.subplot(len(image), 1, i)
        plt.imshow(image[i-1])
    plt.show()


