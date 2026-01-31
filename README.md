# DocuTranslate

A document translation tool for Dutch to English that uses OCR and Google Gemini for fast, high-quality translation.

## What It Does

DocuTranslate extracts text from scanned documents (PDFs and images) using Optical Character Recognition (OCR) and translates that text from Dutch to English using Google's Gemini API.

The workflow is straightforward:

1. Input a document (PDF, PNG, JPG, or JPEG)
2. Tesseract OCR extracts the Dutch text from the document
3. Google Gemini translates the text to English
4. Output is saved as a plain text file

## Privacy Notice

**This tool sends document text to Google's servers for translation.** If you have sensitive documents that must remain private, consider using an offline translation solution instead.

## API Key Setup

DocuTranslate requires a Google Gemini API key:

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Set the environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

The tool also accepts `GOOGLE_API_KEY` as an alternative environment variable name.

## Prerequisites

### Option 1: Docker (Recommended)

Docker provides the easiest setup with all dependencies pre-configured:

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- A valid Gemini API key

### Option 2: Local Python Installation

For running without Docker, you need:

- **Python 3.11+**
- **Tesseract OCR** with Dutch language support
- **poppler-utils** for PDF processing

#### macOS

```bash
# Install Tesseract with Dutch language pack
brew install tesseract tesseract-lang

# Install poppler for PDF support
brew install poppler
```

#### Ubuntu/Debian

```bash
# Install Tesseract with Dutch language pack
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-nld

# Install poppler for PDF support
sudo apt-get install poppler-utils
```

#### Windows

1. Download and install Tesseract from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. During installation, select the Dutch language pack
3. Download poppler from [poppler releases](https://github.com/osber/poppler-windows/releases) and add to PATH

## Usage

### Using Docker (Recommended)

Build the Docker image:

```bash
docker build -t docu-translate .
```

Translate a document:

```bash
# Mount your documents directory and run
docker run --rm \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -v /path/to/your/documents:/documents \
  docu-translate /documents/your-file.pdf
```

The translated output will be saved alongside the original file with `_translated.txt` appended.

Example:

```bash
# Translate a bank letter
docker run --rm \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -v ~/Documents:/documents \
  docu-translate /documents/bank_letter.pdf

# Output: /documents/bank_letter_translated.txt
```

### Using Docker Compose

First, set your API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

Then run:

```bash
# Build the service
docker-compose build

# Translate a document
docker-compose run --rm app /documents/your-file.pdf
```

### Local Python Usage

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Set your API key and run the translation:

```bash
export GEMINI_API_KEY="your-key-here"
python main.py /path/to/your/document.pdf
```

### Web Interface (Recommended for Non-Technical Users)

DocuTranslate includes a simple web UI for drag-and-drop document translation:

```bash
# Set your API key
export GEMINI_API_KEY="your-key-here"

# Start the web interface
python app.py
```

This opens a browser at `http://localhost:7860` where you can:
- Drag and drop files to upload
- Select the translation model
- View original and translated text side by side
- Download the translation

### Model Selection

You can choose different Gemini models based on your needs:

```bash
# Default (fast, good quality)
python main.py document.pdf

# Faster model
python main.py --model gemini-2.0-flash document.pdf

# Higher quality (slower, may have stricter rate limits)
python main.py --model gemini-2.5-pro document.pdf
```

Available models:
- `gemini-2.0-flash` (default) - Fast and efficient
- `gemini-2.5-flash` - Balanced speed and quality
- `gemini-2.5-pro` - Highest quality

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Multi-page documents supported; each page is processed sequentially |
| PNG | `.png` | Single image files |
| JPEG | `.jpg`, `.jpeg` | Single image files |

## Example Output

**Original Dutch Document (bank letter excerpt):**

```
Geachte heer/mevrouw,

Hierbij informeren wij u over de wijzigingen in uw spaarrekening.
Per 1 januari 2024 wordt de rente aangepast naar 1,5% per jaar.
Uw huidige saldo bedraagt EUR 12.450,00.

Met vriendelijke groet,
ABN AMRO Bank N.V.
```

**Translated English Output:**

```
Dear Sir/Madam,

We hereby inform you about the changes to your savings account.
As of January 1, 2024, the interest rate will be adjusted to 1.5% per year.
Your current balance is EUR 12,450.00.

Kind regards,
ABN AMRO Bank N.V.
```

## Performance

DocuTranslate with Gemini is significantly faster than local translation models:

| Phase | Time (4-page document) |
|-------|------------------------|
| PDF conversion | ~0.5s |
| OCR extraction | ~3-4 min |
| Translation | ~5-10s |
| **Total** | **~3-4 min** |

### Comparison with Local Models

| Metric | Gemini | Helsinki-NLP (local) |
|--------|--------|----------------------|
| Translation time (4-page) | ~5-10s | ~3 min |
| Docker image size | ~500MB | ~3GB |
| First-run setup | None | Download 300MB model |
| Requires internet | Yes | No (after setup) |

### Optimizations Applied

- Parallel OCR processing (4 workers)
- Optimized Tesseract configuration (`--psm 6 --oem 1`)
- Image preprocessing for better OCR accuracy
- Automatic text chunking for large documents
- Exponential backoff retry on rate limits

## Rate Limits

Gemini API has rate limits that vary by model and account type. If you hit rate limits, the tool will automatically retry with exponential backoff. For heavy usage, consider:

1. Using the free tier API key for testing, then upgrading for production
2. Selecting a faster model (`gemini-2.0-flash`) which typically has higher rate limits
3. Processing documents in smaller batches

## License

MIT License
