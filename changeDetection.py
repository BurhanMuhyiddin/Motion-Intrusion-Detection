# Burhan Mohayaddin

import cv2
import numpy as np


class DetectChange:
    def __init__(self):
        # parameter for thresholding and background update
        self.threshold = 50
        self.alpha = 0.15

        self.bg_interpolated = None  # reference background image
        # buffer to keep image samples for background image extraction
        self.background_buffer = []
        self.num_bck_imgs = 100  # number of images to store for background image extraction

        #   countour colors (red :  person
        #                    blue:  true object
        #                    green: false object)
        self.cnt_colors = [(255, 0, 0), (0, 255, 0),
                           (0, 0, 255)]
        self.table = []  # table to store object datas

        self.cap = cv2.VideoCapture("video.wm")
        self.out = open("Result.txt", "w")

    def start(self):

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if ret:
                frame = frame.astype(float)

                # wait until first 100 frames to be collected
                # then estimate background image
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) <= self.num_bck_imgs:
                    self.background_buffer.append(frame)
                    if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.num_bck_imgs:
                        self.generate_background()
                # after background extracted detect motion
                else:
                    # get mask by finding l1 distance
                    # before finding distance, both reference and traget images have been
                    # filtered by Gaussian filter to smooth images and remove small unrelated details
                    mask = (
                        self.l1_distance(
                            frame, self.bg_interpolated) > self.threshold
                    )

                    # contours have been found on mask and contours that have small area
                    # have been removed and new mask has been created
                    contours = cv2.findContours(
                        np.uint8(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
                    temp_mask = np.ones(mask.shape[:2], dtype=np.uint8)
                    for cnt in contours:
                        if cv2.contourArea(cnt) < 350:
                            cv2.drawContours(temp_mask, [cnt], 0, 0, -1)
                    new_mask = cv2.bitwise_and(
                        np.uint8(mask), np.uint8(mask), mask=temp_mask)

                    # apply morphological operations to improve mask
                    # close holes and get rid of noise
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (3, 3))
                    new_mask = cv2.morphologyEx(np.uint8(
                        new_mask), cv2.MORPH_CLOSE, kernel, iterations=1)
                    new_mask = cv2.morphologyEx(np.uint8(
                        mask), cv2.MORPH_OPEN, kernel, iterations=1)
                    new_mask = cv2.dilate(
                        np.uint8(new_mask), (15, 15), iterations=2)

                    # update background image
                    self.bg_interpolated[new_mask == 0] = (
                        self.alpha * frame[new_mask == 0]
                        + (1 - self.alpha) *
                        self.bg_interpolated[new_mask == 0]
                    )

                    # get final contours by filtering them
                    # store their detail
                    num_detected_objects = 0
                    contours = cv2.findContours(
                        np.uint8(new_mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
                    contours_data = []
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 500:
                            perimeter = cv2.arcLength(cnt, True)
                            compactness = perimeter * perimeter / area
                            M = cv2.moments(cnt)
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])
                            data = (area, cnt, perimeter, compactness, cx, cy)
                            contours_data.append(data)

                    # sort contour by areas which will easen
                    # the process of eliminating intersected contours (bounding rectangle has been obtained
                    # and checked whether they intersect in order to get rid of multiple contours detected
                    # for the same object)
                    # and the process of catogerizing objects.
                    # Objects are categorized based on their areas, perimeters
                    # and contour approximate shape
                    contours_data = sorted(
                        contours_data, key=lambda x: x[0], reverse=True)
                    flags = np.ones(len(contours_data))
                    obj_type = ""
                    obj_types = []
                    for i in range(len(contours_data)):
                        if flags[i] != 0:
                            num_detected_objects += 1
                            color = None
                            obj_type = None
                            cmpts = contours_data[i][3]
                            if 15.0 <= cmpts <= 17.0:
                                color = self.cnt_colors[1]
                                obj_type = "false object"
                            elif len(contours_data) == 1 or contours_data[i][0] > 3000:
                                color = self.cnt_colors[0]
                                obj_type = "person"
                            else:
                                color = self.cnt_colors[2]
                                obj_type = "true object"

                            obj_types.append(obj_type)
                            cv2.drawContours(
                                frame, [contours_data[i][1]], -1, color, 2)
                            bRi = cv2.boundingRect(contours_data[i][1])
                            for j in range(i+1, len(contours_data)):
                                bRj = cv2.boundingRect(contours_data[j][1])
                                if self.does_intersect(bRi, bRj):
                                    flags[j] = 0

                    temp = []
                    temp.append(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    temp.append(num_detected_objects)
                    temp.append(obj_types)

                    for i in np.where(flags == 1)[0]:
                        temp.append(contours_data[i])

                    self.table.append(temp)

                    cv2.imshow(
                        "Annotated Video", cv2.cvtColor(frame.astype(
                            np.uint8), cv2.COLOR_BGR2RGB)
                    )
            else:
                break

            if cv2.waitKey(40) & 0xFF == ord("q"):
                self.write_to_file()
                break
        self.write_to_file()

    def write_to_file(self):
        for row in self.table:
            line = str(int(row[0])) + "\t\t" + str(row[1]) + "\n"
            for i in range(int(row[1])):
                line = line + str(i+1) + "\t\t" + \
                    str(row[3+i][0]) + "\t\t" + str(row[3+i][2]
                                                    ) + "\t\t" + str(row[3+i][3]) + "\t\t" + "(" + str(row[3+i][4]) + "," + str(str(row[3+i][5])) + ")" + "\t\t" + row[2][i] + "\n"
            self.out.write(line)
            self.out.write(
                "-----------------------------------------------------------------------\n\n")

    def does_intersect(self, a, b):
        # by multiplying alpha increase rectangle sizes a bit
        # so nearby rectangles will intersect and we can eliminate them
        alpha = 1.5
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+alpha*a[2], b[0]+alpha*b[2]) - x
        h = min(a[1]+alpha*a[3], b[1]+alpha*b[3]) - y
        if w < 0 or h < 0:
            return False
        return True

    def generate_background(self):
        self.bg_interpolated = np.stack(self.background_buffer, axis=0)
        self.bg_interpolated = np.median(self.bg_interpolated, axis=0)

    def l1_distance(self, img1, img2):
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
        img2 = cv2.GaussianBlur(img2, (5, 5), 0)

        diff = np.abs(img1 - img2)
        if img1.shape[-1] == 3 and len(img1.shape) == 3:
            diff = np.sum(diff, axis=-1)
        return diff

    def display_image_and_mask(self, img, mask, label):
        img_copy = img.copy()
        img_copy[np.logical_not(mask)] = np.asarray([255, 255, 255])
        cv2.imshow(
            label, cv2.cvtColor(
                img_copy.astype(np.uint8), cv2.COLOR_BGR2RGB)
        )

    def __del__(self):
        cv2.destroyAllWindows()
        self.out.close()


def main():
    dc = DetectChange()
    dc.start()


if __name__ == "__main__":
    main()
