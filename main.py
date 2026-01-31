#!/usr/bin/env python3
"""
DocuTranslate - Dutch to English Document Translation

This script extracts text from PDF or image files using Tesseract OCR
and translates the Dutch text to English using Google Gemini.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image, ImageEnhance
from pdf2image import convert_from_path
from google import genai
from google.genai import types


# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg'}

# Default Gemini model
DEFAULT_MODEL = "gemini-2.0-flash"

# Tesseract OCR configuration for optimal speed
# --psm 6: Assume single uniform block of text (faster than auto-detect)
# --oem 1: LSTM engine only (best speed/accuracy balance)
TESSERACT_CONFIG = '--psm 6 --oem 1'

# Number of parallel workers for OCR processing
OCR_WORKERS = 4


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate Dutch documents (PDF/images) to English",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py document.pdf
  python main.py image.png
  python main.py --model gemini-2.5-pro /path/to/dutch_letter.jpg
        """
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the document file (PDF, PNG, JPG, or JPEG)"
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL}). Options: gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro"
    )
    return parser.parse_args()


def validate_file(file_path: str) -> Path:
    """
    Validate that the file exists and has a supported extension.

    Args:
        file_path: Path to the file to validate

    Returns:
        Path object for the validated file

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file extension is not supported
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file extension: {extension}. "
            f"Supported extensions: {supported}"
        )

    return path


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image for optimal OCR performance.

    Applies grayscale conversion, contrast enhancement, and binarization
    to improve OCR speed and accuracy.

    Args:
        image: PIL Image object

    Returns:
        Preprocessed PIL Image
    """
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')

    # Enhance contrast for better text recognition
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    # Binarize with threshold (black text on white background)
    image = image.point(lambda p: 255 if p > 128 else 0)

    return image


def extract_text_from_image(image: Image.Image, page_num: Optional[int] = None) -> str:
    """
    Extract text from an image using Tesseract OCR.

    Args:
        image: PIL Image object
        page_num: Optional page number for logging (1-indexed)

    Returns:
        Extracted text from the image
    """
    page_info = f" (page {page_num})" if page_num else ""
    print(f"  - Extracting text{page_info}...")

    # Preprocess image for better OCR performance
    processed_image = preprocess_image_for_ocr(image)

    # Configure Tesseract for Dutch language with optimized settings
    # Using 'nld' for Dutch (Nederlands)
    text = pytesseract.image_to_string(
        processed_image,
        lang='nld',
        config=TESSERACT_CONFIG
    )
    return text.strip()


def _ocr_single_page(args: tuple[Image.Image, int]) -> tuple[int, str]:
    """
    OCR a single page (helper for parallel processing).

    Args:
        args: Tuple of (image, page_number)

    Returns:
        Tuple of (page_number, extracted_text)
    """
    image, page_num = args

    # Preprocess image for better OCR performance
    processed_image = preprocess_image_for_ocr(image)

    # Run OCR with optimized settings
    text = pytesseract.image_to_string(
        processed_image,
        lang='nld',
        config=TESSERACT_CONFIG
    )
    return page_num, text.strip()


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file by converting pages to images and OCR.

    Uses parallel processing for faster OCR on multi-page documents.

    Args:
        file_path: Path to the PDF file

    Returns:
        Combined text from all pages
    """
    print(f"Converting PDF to images...")
    pdf_start = time.time()

    try:
        # Optimized pdf2image settings:
        # - dpi=200: Good balance of speed vs quality for OCR
        # - grayscale=True: Faster conversion, OCR only needs grayscale
        # - thread_count=4: Parallel PDF rendering
        images = convert_from_path(
            str(file_path),
            dpi=200,
            grayscale=True,
            thread_count=4
        )
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to images: {e}")

    print(f"  PDF conversion took {time.time() - pdf_start:.1f}s")

    total_pages = len(images)
    print(f"Found {total_pages} page(s) in PDF")

    # Use parallel processing for OCR
    print(f"Running OCR with {OCR_WORKERS} parallel workers...")
    ocr_start = time.time()
    results: dict[int, str] = {}

    # Prepare arguments for parallel processing
    page_args = [(img, i) for i, img in enumerate(images, start=1)]

    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as executor:
        # Submit all OCR tasks
        futures = {
            executor.submit(_ocr_single_page, args): args[1]
            for args in page_args
        }

        # Collect results as they complete
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                _, page_text = future.result()
                results[page_num] = page_text
                print(f"  - Completed page {page_num}/{total_pages}")
            except Exception as e:
                print(f"  - Warning: Failed to OCR page {page_num}: {e}")
                results[page_num] = ""

    print(f"  OCR took {time.time() - ocr_start:.1f}s")

    # Reconstruct text in page order
    all_text = []
    for i in range(1, total_pages + 1):
        page_text = results.get(i, "")
        if page_text:
            all_text.append(f"--- Page {i} ---\n{page_text}")

    return "\n\n".join(all_text)


def extract_text_from_image_file(file_path: Path) -> str:
    """
    Extract text from an image file using OCR.

    Args:
        file_path: Path to the image file

    Returns:
        Extracted text from the image
    """
    print(f"Loading image...")

    try:
        image = Image.open(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {e}")

    return extract_text_from_image(image)


def extract_text(file_path: Path) -> str:
    """
    Extract text from a document file (PDF or image).

    Args:
        file_path: Path to the document file

    Returns:
        Extracted text from the document
    """
    print(f"\n[Step 1/2] Extracting text from: {file_path.name}")

    extension = file_path.suffix.lower()

    if extension == '.pdf':
        return extract_text_from_pdf(file_path)
    else:
        return extract_text_from_image_file(file_path)


def get_gemini_client() -> genai.Client:
    """
    Initialize Gemini client with API key from environment.

    Returns:
        Configured Gemini client

    Raises:
        ValueError: If no API key is found
    """
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError(
            "Gemini API key not found.\n"
            "Set GEMINI_API_KEY environment variable.\n"
            "Get your key at: https://aistudio.google.com/app/apikey"
        )
    return genai.Client(api_key=api_key)


def split_into_chunks(text: str, max_chars: int = 4000) -> list[str]:
    """
    Split text into chunks at paragraph boundaries.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current:
        chunks.append(current.strip())

    return chunks


def translate_with_retry(
    client: genai.Client,
    model: str,
    text: str,
    config: types.GenerateContentConfig,
    max_retries: int = 3
) -> str:
    """
    Translate text with exponential backoff on rate limits.

    Args:
        client: Gemini client
        model: Model name to use
        text: Text to translate
        config: Generation configuration
        max_retries: Maximum number of retry attempts

    Returns:
        Translated text

    Raises:
        RuntimeError: If max retries exceeded
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=text,
                config=config
            )
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "quota" in error_str or "rate" in error_str:
                delay = 2 ** attempt
                print(f"  - Rate limited, waiting {delay}s...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Max retries exceeded due to rate limiting")


def translate_text(text: str, model: str = DEFAULT_MODEL) -> str:
    """
    Translate Dutch text to English using Gemini.

    Args:
        text: Dutch text to translate
        model: Gemini model to use

    Returns:
        Translated English text
    """
    print(f"\n[Step 2/2] Translating text using {model}...")
    translate_start = time.time()

    if not text.strip():
        print("  - Warning: No text to translate")
        return ""

    client = get_gemini_client()

    # Configure the translation prompt
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a professional Dutch to English translator. "
            "Translate the following Dutch text to English. "
            "Preserve formatting including line breaks. "
            "Output only the translation, no commentary."
        ),
        temperature=0.3,
    )

    # Handle large documents by chunking
    chunks = split_into_chunks(text, max_chars=4000)
    translated_chunks = []
    total_chunks = len(chunks)

    print(f"  - Processing {total_chunks} chunk(s)...")

    for i, chunk in enumerate(chunks, start=1):
        print(f"  - Translating chunk {i}/{total_chunks}...")

        translated = translate_with_retry(client, model, chunk, config)
        translated_chunks.append(translated)

    print(f"  - Translation complete in {time.time() - translate_start:.1f}s")
    return "\n\n".join(translated_chunks)


def save_output(
    file_path: Path,
    original_text: str,
    translated_text: str
) -> Path:
    """
    Save the original and translated text to a file.

    Args:
        file_path: Original document path
        original_text: Original Dutch text
        translated_text: Translated English text

    Returns:
        Path to the output file
    """
    # Create output filename
    output_path = file_path.parent / f"{file_path.stem}_translated.txt"

    content = f"""{'='*60}
DOCUTRANSLATE - Dutch to English Translation
{'='*60}
Source file: {file_path.name}
{'='*60}

{'='*60}
ORIGINAL DUTCH TEXT
{'='*60}

{original_text}

{'='*60}
ENGLISH TRANSLATION
{'='*60}

{translated_text}
"""

    try:
        output_path.write_text(content, encoding='utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to save output file: {e}")

    return output_path


def main() -> int:
    """
    Main entry point for DocuTranslate.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("="*60)
    print("DocuTranslate - Dutch to English Document Translation")
    print("="*60)

    try:
        # Parse arguments
        args = parse_arguments()

        # Validate file
        file_path = validate_file(args.file_path)
        print(f"\nInput file: {file_path}")
        print(f"Model: {args.model}")

        # Extract text using OCR
        original_text = extract_text(file_path)

        if not original_text.strip():
            print("\nError: No text could be extracted from the document.")
            print("Please ensure the document contains readable text.")
            return 1

        # Translate text
        translated_text = translate_text(original_text, args.model)

        # Save output
        output_path = save_output(file_path, original_text, translated_text)

        # Print results
        print("\n" + "="*60)
        print("ORIGINAL DUTCH TEXT")
        print("="*60)
        print(original_text)

        print("\n" + "="*60)
        print("ENGLISH TRANSLATION")
        print("="*60)
        print(translated_text)

        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)
        print(f"Output saved to: {output_path}")

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except ValueError as e:
        print(f"\nError: {e}")
        return 1
    except RuntimeError as e:
        print(f"\nError: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
