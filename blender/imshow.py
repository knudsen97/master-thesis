import cv2
import matplotlib.pyplot as plt

def main():
    image = cv2.imread("tilt.png", cv2.IMREAD_ANYDEPTH)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()