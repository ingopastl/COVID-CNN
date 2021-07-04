from pathlib import Path
import imageio
import imgaug.augmenters as iaa
import natsort


def read_images(path):
    directories = list(path.iterdir())
    directories_sorted = []
    for fn in sorted([str(p) for p in directories]):
        directories_sorted.append(fn)
    directories_sorted = natsort.natsorted(directories, key=str)
    return directories_sorted


def augment():
    comp_functions = [
        (iaa.Fliplr(1), 'fliplr'),
        (iaa.GaussianBlur(sigma=(0.0, 3.0)), 'blur_gaussian'),
        (iaa.LinearContrast((0.4, 1.6)), 'contrast_linear'),
        (iaa.AdditiveLaplaceNoise(scale=(0, 0.2 * 255)), 'add_laplace'),
        (iaa.Multiply((0.5, 1.5), per_channel=0.5), 'mult')
    ]

    functions = comp_functions

    print("Augmentation functions loaded...")

    # Read images and coordinates from .jpg's and .txt's
    path_name = r'Covid19-dataset\train'
    path = Path(path_name)
    directories = list(path.iterdir())

    for directory in directories:
        images = read_images(directory)

        image_files = []
        for image in images:
            image_files.append(image)

        print("Executing functions...")

        # Run agumentation for isolated functions
        for i, image_path in enumerate(image_files):
            image = imageio.imread(str(image_path))
            print(image_path)

            for function in functions:
                seq = iaa.Sequential([function[0]])
                for j in range(0, 1):
                    image_aug = seq.augment_image(image=image)
                    imageio.imwrite(str(directory) + '\\' + str(i) + '_' + function[1] + str(j) + '.png', image_aug)

    print("Finished")


if __name__ == '__main__':
    augment()
