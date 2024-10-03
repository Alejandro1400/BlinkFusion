from ridge_detection.lineDetector import LineDetector
from ridge_detection.params import Params, load_json
from PIL import Image
from mrcfile import open as mrcfile_open

def detect_ridges(image_path, config_file):
    # Load the configuration
    config = load_json(config_file)
    # Update the image path in the configuration
    config['path_to_file'] = image_path
    params = Params(config_file)

    # Load the image
    try:
        img = mrcfile_open(image_path).data if image_path.endswith('.mrc') else Image.open(image_path)
    except Exception as e:
        raise IOError(f"Failed to load image: {e}")

    # Initialize the line detector
    detect = LineDetector(params=config_file)
    
    # Perform line detection
    result = detect.detectLines(img)
    resultJunction = detect.junctions

    # Return the detection results
    return result, resultJunction