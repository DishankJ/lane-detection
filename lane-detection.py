import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*2.8/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)
    right_line = right_line.reshape(2,2)
    right_line = np.flip(right_line, 0)
    right_line = right_line.reshape(4)
    return np.array([left_line, right_line]), left_fit_avg, right_fit_avg
        
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    canny = cv2.Canny(blur, 75, 150)
    return canny

def bitmap_generation(image, averaged_lines, left_line_parameters):
    polygon = [[0,image.shape[0]]]
    for x1, y1, x2, y2 in averaged_lines:
        if x1 < 0:
            x1 = 0
            slope, intercept = left_line_parameters
            y1 = int(intercept)
        polygon.append([x1,y1])
        polygon.append([x2,y2])
    bitmap = 255 + np.zeros_like(image)
    lane_region = np.zeros_like(image)
    polygons = np.array([polygon])
    cv2.fillPoly(bitmap, polygons, color=(0,0,0))
    cv2.fillPoly(lane_region, polygons, color=(255,0,0))
    return bitmap, lane_region

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) # the last argument is line thickness
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[[0, 500], [453, 300], [605, height], [0, height]]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, color=(255,0,0))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image, mask

image = cv2.imread('testimg.jpg')
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image, mask = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines, left_line_parameters, right_line_parameters = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

bitmap, lane_region = bitmap_generation(image, averaged_lines, left_line_parameters)

isolated_image = cv2.bitwise_and(image, lane_region)

cv2.imshow('lane detection', combo_image)
cv2.imshow('bitmap generation', bitmap)
cv2.imshow('background isolated image', isolated_image)
cv2.waitKey(0)
# plt.imshow(bitmap)
# plt.show()