import sys
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import cv2
import os
import shutil
from difflib import SequenceMatcher
from PyPDF2 import PdfReader
import json
import logging
import argparse
import hashlib
from openai import OpenAI

# Constants
ORANGE_LOWER_BOUND = np.array([0, 120, 240])  # Valid lower bound BGR
ORANGE_UPPER_BOUND = np.array([239, 247, 255])  # Valid upper bound BGR
BLUE_LOWER_BOUND = np.array([230, 115, 0])  # Valid lower bound BGR
BLUE_UPPER_BOUND = np.array([255, 238, 218])  # Valid upper bound BGR
KERNEL_SIZE = (35, 35)  # Kernel size for morphological operations
EXPAND_BY = 10  # Number of pixels to expand bounding boxes by
SIMILARITY_THRESHOLD = 0.7  # Threshold for determining similar texts

# Setup argument parser
parser = argparse.ArgumentParser(description="PDF Difference Analyzer")
parser.add_argument('--log-level', type=str, default='INFO',
                    help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
args = parser.parse_args()

# Setup logging
logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                    format='\033[92m[%(asctime)s] %(levelname)s: %(message)s\033[0m',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Check Python path
logging.debug(f"Python executable: {sys.executable}")
logging.debug(f"Python version: {sys.version}")
logging.debug(f"Python path: {sys.path}")

logging.debug("Tesseract imported successfully!")

# Validate and initialize OpenAI API
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("The OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=openai_api_key)

# Function to clear and create output directories
def setup_output_directories(base_dirs):
    """
    Clears and creates the necessary output directories for storing masks, contours, regions, and pages.
    
    Args:
    base_dirs (list): List of base directory paths to setup.
    """
    for base_dir in base_dirs:
        for sub_dir in ["masks", "contours", "regions", "pages"]:
            dir_path = os.path.join(base_dir, sub_dir)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)
    # Create additional directory for diff_pages
    if os.path.exists("diff_pages"):
        shutil.rmtree("diff_pages")
    os.makedirs("diff_pages")

# Function to convert PDF pages to images
def convert_pdf_to_images(pdf_path, output_folder):
    """
    Converts a PDF document to images, with each page saved as a separate image file.
    
    Args:
    pdf_path (str): Path to the PDF file.
    output_folder (str): Folder to save the output images.

    Returns:
    list: List of images corresponding to each page of the PDF.
    """
    logging.debug(f"Converting PDF to images: {pdf_path}")
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        image.save(os.path.join(output_folder, f"page_{i + 1}.png"))
    logging.debug(f"Converted {len(images)} pages and saved to {output_folder}.")
    return images

# Function to detect colored regions
def detect_colored_regions(image, color, page_num, base_dir):
    """
    Detects colored regions in an image and saves masks and closed masks for verification.
    
    Args:
    image (PIL.Image.Image): Image to process.
    color (str): Color to detect ('orange' or 'blue').
    page_num (int): Page number of the image.
    base_dir (str): Base directory to save masks and contours.

    Returns:
    list: List of contours detected in the image.
    """
    logging.debug(f"Detecting {color} regions on page {page_num}.")
    # Convert image to numpy array
    img_np = np.array(image)
    logging.debug(f"Image converted to numpy array.")

    # Convert image from RGB to BGR (OpenCV uses BGR)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    logging.debug(f"Image converted from RGB to BGR.")

    # Define the color range for detection
    if color == 'orange':
        lower_bound = ORANGE_LOWER_BOUND
        upper_bound = ORANGE_UPPER_BOUND
    elif color == 'blue':
        lower_bound = BLUE_LOWER_BOUND
        upper_bound = BLUE_UPPER_BOUND
    else:
        raise ValueError("Color not supported")

    logging.debug(f"Using lower bound {lower_bound} and upper bound {upper_bound} for color detection.")

    # Create mask
    mask = cv2.inRange(img_np, lower_bound, upper_bound)
    logging.debug(f"Mask created. Saving mask for verification.")
    
    # Save mask for verification
    mask_image = Image.fromarray(mask)
    mask_image_path = os.path.join(base_dir, "masks", f"mask_page_{page_num}.png")
    mask_image.save(mask_image_path)
    logging.debug(f"Saved mask to {mask_image_path}")

    # Apply morphological closing to merge nearby regions
    kernel = np.ones(KERNEL_SIZE, np.uint8)  # Adjust kernel size based on distance between letters
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Save closed mask for verification
    closed_mask_image = Image.fromarray(closed_mask)
    closed_mask_image_path = os.path.join(base_dir, "masks", f"closed_mask_page_{page_num}.png")
    closed_mask_image.save(closed_mask_image_path)
    logging.debug(f"Saved closed mask to {closed_mask_image_path}")

    # Find contours
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging.debug(f"Found {len(contours)} contours.")
    
    return contours

# Function to expand bounding boxes
def expand_bounding_box(x, y, w, h, expand_by, image_width, image_height):
    """
    Expands bounding boxes by a specified number of pixels, ensuring they stay within image boundaries.
    
    Args:
    x (int): X coordinate of the bounding box.
    y (int): Y coordinate of the bounding box.
    w (int): Width of the bounding box.
    h (int): Height of the bounding box.
    expand_by (int): Number of pixels to expand by.
    image_width (int): Width of the image.
    image_height (int): Height of the image.

    Returns:
    tuple: Expanded bounding box coordinates and size.
    """
    x = max(0, x - expand_by)
    y = max(0, y - expand_by)
    w = min(image_width - x, w + 2 * expand_by)
    h = min(image_height - y, h + 2 * expand_by)
    return x, y, w, h

# Function to perform OCR on an image
def ocr_image(image):
    """
    Performs OCR (Optical Character Recognition) on an image to extract text.
    
    Args:
    image (PIL.Image.Image): Image to perform OCR on.

    Returns:
    str: Extracted text from the image.
    """
    logging.debug("Performing OCR on image.")
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    logging.debug("OCR completed.")
    return text

# Post-process OCR results to correct common mistakes
def postprocess_ocr_text(text):
    """
    Corrects common OCR mistakes in the extracted text.
    
    Args:
    text (str): OCR extracted text.

    Returns:
    str: Corrected text.
    """
    corrections = {
        "Clinvar": "ClinVar"
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

def compare_texts(text1, text2):
    """
    Compares two texts and returns a similarity ratio.
    
    Args:
    text1 (str): First text to compare.
    text2 (str): Second text to compare.

    Returns:
    float: Similarity ratio between the two texts.
    """
    return SequenceMatcher(None, text1.strip(), text2.strip()).ratio()

# Function to extract JSON metadata from a PDF
def extract_metadata(pdf_path):
    """
    Extracts JSON metadata from a PDF file.
    
    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    dict or None: Extracted metadata if available, otherwise None.
    """
    reader = PdfReader(pdf_path)
    metadata = reader.metadata
    if metadata and '/metadata' in metadata:
        raw_metadata = metadata['/metadata']
        try:
            metadata_json = json.loads(raw_metadata)
            return metadata_json
        except json.JSONDecodeError:
            logging.error("Error decoding JSON metadata.")
            return None
    else:
        logging.debug("No metadata found in PDF.")
        return None

def get_cache_filename(query):
    """
    Generates a cache filename based on the hash of the query.
    
    Args:
    query (str): The query string to hash.

    Returns:
    str: The cache filename.
    """
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return os.path.join("cache", f"{query_hash}.json")

def read_cache(query):
    """
    Reads the cached response for a given query.
    
    Args:
    query (str): The query string.

    Returns:
    dict or None: The cached response if available, otherwise None.
    """
    cache_filename = get_cache_filename(query)
    if os.path.exists(cache_filename):
        with open(cache_filename, "r") as cache_file:
            return json.load(cache_file)
    return None

def write_cache(query, response):
    """
    Writes the response to the cache for a given query.
    
    Args:
    query (str): The query string.
    response (dict): The response to cache.
    """
    os.makedirs("cache", exist_ok=True)
    cache_filename = get_cache_filename(query)
    with open(cache_filename, "w") as cache_file:
        json.dump(response, cache_file)

def find_component_name(summary_json, pdf_payload):
    """
    Finds the component name using the OpenAI API with caching.
    
    Args:
    summary_json (list): Summary of changes.
    pdf_payload (dict): Payload data from the PDF metadata.

    Returns:
    list: Updated summary JSON with component names.
    """
    query = f"""
    Here is a summary of PDF diffing script:
    {json.dumps(summary_json)}

    Here is a payload which helped to generate the PDF:
    {json.dumps(pdf_payload)}

    VERY IMPORTANT. Give the answer in JSON format of the the summary json structure described above by replacing "__COMPONENT_PLACEHOLDER__" with the name of the `componentName` involved in the diff. Replace with "Unknown" if you unable to recognize the source component.
    The JSON should be valid and parseable by python's json.loads(...) function
    DO NOT use any formatting.
    """

    # Check cache
    cached_response = read_cache(query)
    if cached_response:
        logging.debug("Returning cached response.")
        return cached_response

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": query}
        ],
        temperature=0
    )

    # Extract the JSON response
    response_json = response.choices[0].message.content.strip().replace("\n", "")
    logging.debug(response_json)
    response_data = json.loads(response_json)

    # Write to cache
    write_cache(query, response_data)

    return response_data

def analyze_differences(diff_pdf, baseline_pdf, changed_pdf):
    """
    Analyzes the differences between the baseline and changed PDFs by detecting and comparing regions with differences.

    Args:
    diff_pdf (str): Path to the diff PDF.
    baseline_pdf (str): Path to the baseline PDF.
    changed_pdf (str): Path to the changed PDF.

    Returns:
    list: Summary of changes with component names.
    """
    # Setup output directories
    setup_output_directories(["baseline", "changed"])

    # Extract metadata from baseline PDF
    baseline_metadata = extract_metadata(baseline_pdf)
    if baseline_metadata is None:
        logging.debug("No metadata found in baseline PDF.")
        return
    payload = baseline_metadata["payload"]

    # Convert diff.pdf to images
    diff_images = convert_pdf_to_images(diff_pdf, "diff_pages")

    # Convert baseline.pdf to images
    baseline_images = convert_pdf_to_images(baseline_pdf, os.path.join("baseline", "pages"))

    # Convert changed.pdf to images
    changed_images = convert_pdf_to_images(changed_pdf, os.path.join("changed", "pages"))

    changes = []
    baseline_texts = []
    changed_texts = []
    baseline_contours = []
    changed_contours = []

    for page_num, diff_image in enumerate(diff_images):
        logging.debug(f"Processing page {page_num + 1}/{len(diff_images)}")

        image_width, image_height = diff_image.size

        # Detect orange regions
        orange_contours = detect_colored_regions(diff_image, 'orange', page_num + 1, "changed")
        logging.debug(f"Merged to {len(orange_contours)} orange contours.")

        for rect_num, cnt in enumerate(orange_contours):
            x, y, w, h = cv2.boundingRect(cnt)
            x, y, w, h = expand_bounding_box(x, y, w, h, EXPAND_BY, image_width, image_height)
            logging.debug(f"Orange Rect {rect_num + 1}: Expanded bounding box (x={x}, y={y}, w={w}, h={h})")
            roi = changed_images[page_num].crop((x, y, x + w, y + h))
            
            # Save the detected region to disk
            region_path = os.path.join("changed", "regions", f"page_{page_num + 1}_region_{rect_num + 1}.png")
            roi.save(region_path)
            logging.debug(f"Saved detected region to {region_path}")

            # Save the merged region to disk
            merged_region_path = os.path.join("changed", "contours", f"page_{page_num + 1}_merged_region_{rect_num + 1}.png")
            roi.save(merged_region_path)
            logging.debug(f"Saved merged region to {merged_region_path}")

            # Perform OCR on the detected region
            orange_text = ocr_image(roi)
            orange_text = postprocess_ocr_text(orange_text)
            changed_texts.append((page_num + 1, orange_text))
            changed_contours.append((x, y, w, h))
            logging.debug(f"Extracted orange text: {orange_text}")

        # Detect blue regions
        blue_contours = detect_colored_regions(diff_image, 'blue', page_num + 1, "baseline")
        logging.debug(f"Merged to {len(blue_contours)} blue contours.")

        for rect_num, cnt in enumerate(blue_contours):
            x, y, w, h = cv2.boundingRect(cnt)
            x, y, w, h = expand_bounding_box(x, y, w, h, EXPAND_BY, image_width, image_height)
            logging.debug(f"Blue Rect {rect_num + 1}: Expanded bounding box (x={x}, y={y}, w={w}, h={h})")
            roi = baseline_images[page_num].crop((x, y, x + w, y + h))
            
            # Save the detected region to disk
            region_path = os.path.join("baseline", "regions", f"page_{page_num + 1}_region_{rect_num + 1}.png")
            roi.save(region_path)
            logging.debug(f"Saved detected region to {region_path}")

            # Save the merged region to disk
            merged_region_path = os.path.join("baseline", "contours", f"page_{page_num + 1}_merged_region_{rect_num + 1}.png")
            roi.save(merged_region_path)
            logging.debug(f"Saved merged region to {merged_region_path}")

            # Perform OCR on the detected region
            blue_text = ocr_image(roi)
            blue_text = postprocess_ocr_text(blue_text)
            baseline_texts.append((page_num + 1, blue_text))
            baseline_contours.append((x, y, w, h))
            logging.debug(f"Extracted blue text: {blue_text}")

    # Analyze differences
    for i, ((baseline_page_num, baseline_text), (changed_page_num, changed_text)) in enumerate(zip(baseline_texts, changed_texts)):
        similarity_ratio = compare_texts(baseline_text, changed_text)
        baseline_contour = baseline_contours[i]
        changed_contour = changed_contours[i]
        offset = {
            "x_offset": changed_contour[0] - baseline_contour[0],
            "y_offset": changed_contour[1] - baseline_contour[1]
        }
        if similarity_ratio == 1.0:
            change_type = "style change"
        elif similarity_ratio >= SIMILARITY_THRESHOLD:
            change_type = "wording change"
        else:
            change_type = "content change"

        changes.append({
            "page_num": baseline_page_num,  # or changed_page_num since they should be the same
            "baseline_text": baseline_text.replace("\n", " ").strip(),
            "changed_text": changed_text.replace("\n", " ").strip(),
            "type": change_type,
            "offset": offset if change_type == "style change" else None,
            "component_name": "__COMPONENT_PLACEHOLDER__"
        })

    # Call OpenAI API to determine component names
    updated_changes_summary = find_component_name(changes, payload)

    # Print the final summary
    print(json.dumps(updated_changes_summary, indent=4))

    return updated_changes_summary

# Example usage
diff_pdf = "diff.pdf"
baseline_pdf = "baseline.pdf"
changed_pdf = "changed.pdf"

try:
    summary = analyze_differences(diff_pdf, baseline_pdf, changed_pdf)
    logging.debug("Summary of changes:")
    logging.debug(summary)
except Exception as e:
    logging.error(f"An error occurred: {e}")

