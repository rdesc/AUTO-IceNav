""" Convenience script to display an image and get image coordinates for user click """
import argparse
from typing import List

import cv2
import numpy as np


class ClickEvent:
    def __init__(self, image):
        self.points = []
        self.image = image

    def __len__(self):
        return len(self.points)

    def reorder(self):
        self.points = sorted(self.points, key=lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2))

    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            self.points.append([x, y])

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.image, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)

            self.reorder()


def display_img_coords(img, num_pts=None) -> List:
    ce = ClickEvent(img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', ce.click_event)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27 or (num_pts is not None and len(ce) >= num_pts):
            break

    cv2.destroyAllWindows()

    return ce.points[:num_pts]


if __name__ == '__main__':
    # reading the image
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='path to image file')
    args = parser.parse_args()

    print('list of points', display_img_coords(cv2.imread(args.image, 1)))
