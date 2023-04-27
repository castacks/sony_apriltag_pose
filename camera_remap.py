
import cv2

def calculate_new_camera_matrix_and_remap_grids(
    original_shape, 
    new_shape, 
    camera_matrix, 
    distortion_coeffs):
    
    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coeffs,
        imageSize=( original_shape[1], original_shape[0] ),
        alpha=0.0,
        newImgSize=( new_shape[1], new_shape[0] ) )
    
    map0, map1 = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_coeffs,
        R=None,
        newCameraMatrix=new_cam_matrix,
        size=( new_shape[1], new_shape[0] ),
        m1type=cv2.CV_32FC1)
    
    return new_cam_matrix, map0, map1

def read_and_preprocess_image(image_path, map0, map1):
    """
    Reads an image and resizes it to 640x360.
    :param image_path: Path to the image file.
    :param map0: NumPy arrah, the map grid needed by cv2.remap.
    :param map1: NumPy arrah, the map grid needed by cv2.remap.
    :return: Preprocessed image.
    """
    # Load the image as-is
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    assert img is not None, f'Could not read image from {image_path}. '

    # Resize the image to 640x360
    # img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
    img = cv2.remap(img, map0, map1, interpolation=cv2.INTER_LINEAR)

    return img