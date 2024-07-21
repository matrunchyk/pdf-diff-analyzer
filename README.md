# PDF Difference Analyzer

The PDF Difference Analyzer is a tool to detect and analyze differences between two PDF documents. It highlights changes and uses OCR to extract and compare texts, leveraging the OpenAI API to identify component names involved in the changes. It also supports logging and caching to optimize API usage.

## Features

- Convert PDF pages to images
- Detect colored regions (orange and blue) to highlight differences
- Perform OCR on the detected regions to extract text
- Compare texts and classify changes as style, wording, or content changes
- Use OpenAI API to determine component names for the changes
- Extract metadata from PDF files to utilize dynamically injected payloads
- Caching of API responses to optimize repeated queries
- Configurable logging levels from the command line

## Prerequisites

- Python 3.12
- Tesseract OCR
- Poppler-utils (for pdf2image)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/matrunchyk/pdf-diff-analyzer
    cd pdf-difference-analyzer
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure Tesseract OCR is installed and available in your PATH:
    - [Tesseract OCR Installation Guide](https://github.com/tesseract-ocr/tesseract)

4. Ensure Poppler-utils is installed:
    - **Ubuntu**: `sudo apt-get install poppler-utils`
    - **MacOS**: `brew install poppler`

## Usage

### Required PDFs

The script requires three PDF files:
1. `baseline.pdf`: The original version of the document.
2. `changed.pdf`: The modified version of the document.
3. `diff.pdf`: The visual diff between the baseline and changed documents.

The `diff.pdf` can be generated with the `diff-pdf` tool using the following command:

```sh
diff-pdf --output-diff diff.pdf -g baseline.pdf changed.pdf
```

The diff-pdf tool can be found at: [diff-pdf GitHub Repository](https://github.com/vslavik/diff-pdf).

### Command Line Arguments

- `--log-level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is `INFO`.

### Example Command

```sh
python script.py --log-level DEBUG
```

## Environment Variables

- `OPENAI_API_KEY`: Set this environment variable with your OpenAI API key.

## Example Usage in Script

```python
# Example usage in a script
diff_pdf = "diff.pdf"
baseline_pdf = "baseline.pdf"
changed_pdf = "changed.pdf"

try:
    summary = analyze_differences(diff_pdf, baseline_pdf, changed_pdf)
    logging.debug("Summary of changes:")
    logging.debug(summary)
except Exception as e:
    logging.error(f"An error occurred: {e}")
```

## Example Output

```json
[
    {
        "page_num": 1,
        "baseline_text": "",
        "changed_text": "Date: 21-Jul-2024",
        "type": "content change",
        "component_name": "AppFooter"
    },
    {
        "page_num": 1,
        "baseline_text": "All right reserved",
        "changed_text": "All right reserved",
        "type": "style change",
        "offset": {
            "x_offset": 5,
            "y_offset": 0
        },
        "component_name": "AppFooter"
    },
    {
        "page_num": 2,
        "baseline_text": "Confidental Note",
        "changed_text": "Confidential Note",
        "type": "wording change",
        "component_name": "AppHeader"
    }
]
```

## Script Breakdown

### Key Functions

- **setup_output_directories(base_dirs)**

  - Clears and creates necessary output directories for storing masks, contours, regions, and pages.

- **convert_pdf_to_images(pdf_path, output_folder)**

  - Converts a PDF document to images, saving each page as a separate image file.

- **detect_colored_regions(image, color, page_num, base_dir)**

  - Detects colored regions in an image and saves masks and closed masks for verification.

- **expand_bounding_box(x, y, w, h, expand_by, image_width, image_height)**

  - Expands bounding boxes by a specified number of pixels, ensuring they stay within image boundaries.

- **ocr_image(image)**

  - Performs OCR (Optical Character Recognition) on an image to extract text.

- **postprocess_ocr_text(text)**

  - Corrects common OCR mistakes in the extracted text.

- **compare_texts(text1, text2)**

  - Compares two texts and returns a similarity ratio.

- **extract_metadata(pdf_path)**

  - Extracts JSON metadata from a PDF file.

- **find_component_name(summary_json, pdf_payload)**

  - Finds the component name using the OpenAI API with caching.

- **analyze_differences(diff_pdf, baseline_pdf, changed_pdf)**

  - Analyzes the differences between the baseline and changed PDFs by detecting and comparing regions with differences.

## Caching

- The script uses a simple file-based caching system to store OpenAI API responses.
- Cached responses are saved in a cache directory.
- Each query's hash is used as the cache filename.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact
For any questions or suggestions, please contact me [via Telegram](https://t.me/matrunchyk).

