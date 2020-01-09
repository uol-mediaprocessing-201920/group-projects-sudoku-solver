import cv2 as cv


def get_opencv_color_conversion_code(source, target):
    return getattr(cv, "COLOR_" + source.upper() + "2" + target.upper())


def apply_opencv_color_conversion(image, source, target, always_copy=False):
    if source is None or target is None or source == target:
        if always_copy:
            return image.copy()
        else:
            return image
    name = "COLOR_" + source.upper() + "2" + target.upper()
    if not hasattr(cv, name):
        raise Exception("Unknown OpenCV color conversion code: " + name)
    code = getattr(cv, name)
    return cv.cvtColor(image, code)
