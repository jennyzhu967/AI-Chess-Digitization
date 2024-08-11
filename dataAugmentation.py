from PIL import Image
import torchvision.transforms.v2 as transforms
import os
from tqdm import tqdm

augment_times = 7 # only need to change this number
ground_truth_path = "C:/Users/ericz/Desktop/IAM Model/Augmentation_Ground_Truth_" + str(augment_times) + ".csv"
# ft = open(ground_truth_path, 'r+')
# ft.truncate(0)

img_directory_path = "C:/Users/ericz/Desktop/IAM Model/Training_Ground_Truth.txt" # file of Majid training data to augment
main_directory = "C:/Users/ericz/Desktop/IAM Model/Augmentation_" + str(augment_times) # folder to save image to
f = open(img_directory_path, "r")

for x in tqdm(f):
    """
    Below is a lot of nitty gritty work to get the image labels and path
    """
    number = 0
    x = x.strip() # removes all the trailing and leading whitespaces (inlcude new line)
    position_path = x.find('png')
    rel_path = x[:position_path+3] # Find the relative path (without ground truth label)
    # img.show() # for visualization purposes

    position_label = x.find('boxes') # returns the first occurence of the word, -1 if not found

    name = x[position_label + 6: position_path+3] # get the name of the image
    game_number = name[:3] # game number
    ground_truth = x[position_path+3:] # getting the ground truth

    # create a sub folder if it does not already exist
    sub_directory = os.path.join(main_directory, game_number).replace("\\", "/")
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)

    # loop for x different augmentated images and for 1 original image
    for number in range(augment_times+1):  # image 0 is the orignal image
        img = Image.open(rel_path)
        if number !=0: # rotate if it is not the original image
            transform = transforms.RandomChoice(
            [transforms.RandomRotation(degrees = (-10, 10)), # rotate
            transforms.RandomAffine(degrees = 0,
                                    shear = (-15, 15)), # shear
            transforms.RandomAffine(degrees=0,
                                    scale = (0.8, 1.2))] # scale (20% smaller, 20% larger) -> Majid paper: -20%, 20%
            )
            img = transform(img) # perform transform
            # img.show() # show rotated image

        augment_number = "Augment_" + str(number)
        path = augment_number + "_" + name  # path to the new augmented image
        # File Name Format: [Augment #]_[Game #]_[Page #]_[move #]_[black/white]
        # Ex: Augment_10_001_0_1_white (Augment Image 10, Game 1, Page 0, Move 1, Black)

        aug_sub_directory = os.path.join(sub_directory, augment_number).replace("\\", "/")
        if not os.path.exists(aug_sub_directory): # create the subdircetory for augmented images in a game
            os.makedirs(aug_sub_directory)

        SAVE_PATH = os.path.join(aug_sub_directory, path).replace("\\", "/")
        # print(f"SAVE PATH: {SAVE_PATH}")
        label = SAVE_PATH + "," + ground_truth + "\n"
        # print(f"Label: {label}")

        # saving work
        with open(ground_truth_path, "a") as fx: # write it on the new groundTruth
            fx.write(label)
        img.save(SAVE_PATH) # save image to SAVE_PATH