import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_contours_from_mask(mask, min_area=5):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list


def get_larger_contour(contours):
    larger_area = 0
    larger_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > larger_area:
            larger_contour = contour
            larger_area = area
    return larger_contour


def black2white(image):
    lo = np.array([0, 0, 0])
    hi = np.array([0, 0, 0])
    mask = cv2.inRange(image, lo, hi)
    image[mask > 0] = (255, 255, 255)
    return image


def process_image(img, n_w=256, n_h=64):
    w, h, _ = img.shape

    new_w = n_h
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    if w < n_h:
        add_zeros = np.full((n_h - w, h, 3), 0)
        img = np.concatenate((img, add_zeros))
        w, h, _ = img.shape

    if h < n_w:
        add_zeros = np.full((w, n_w - h, 3), 0)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h, _ = img.shape

    if h > n_w or w > n_h:
        dim = (n_w, n_h)
        img = cv2.resize(img, dim)
    return img


def get_image_visualization(img, pred_data, fontpath, font_koef=50):
    h, w = img.shape[:2]
    font = ImageFont.truetype(fontpath, int(h / font_koef))
    empty_img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(empty_img)

    for prediction in pred_data["predictions"]:
        polygon = prediction["polygon"]
        pred_text = prediction["text"]
        cv2.drawContours(img, np.array([polygon]), -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(np.array([polygon]))
        draw.text((x, y), pred_text, fill=0, font=font)

    vis_img = np.array(empty_img)
    vis = np.concatenate((img, vis_img), axis=1)
    return vis


def crop_img_by_polygon(img, polygon):
    # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
    pts = np.array(polygon)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y : y + h, x : x + w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    return dst
