# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocuTranslate is a Dutch-to-English document translation tool. It uses Tesseract OCR to extract text from documents and Google Gemini API for translation.

**Note:** This tool sends document text to Google's servers. It is not a privacy-preserving local-only solution.

## Commands

### Docker (recommended)

```bash
# Build the image
docker build -t docu-translate .

# Translate a document
docker run --rm \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -v ./documents:/documents \
  docu-translate /documents/file.pdf

# Using docker-compose
export GEMINI_API_KEY="your-key"
docker-compose run --rm app /documents/file.pdf
```

### Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key and run translation
export GEMINI_API_KEY="your-key"
python main.py /path/to/document.pdf

# With specific model
python main.py --model gemini-2.5-pro /path/to/document.pdf
```

### Check Python syntax

```bash
python3 -m py_compile main.py
```

## Architecture

Single-file CLI application (`main.py`) with this pipeline:

1. **Input validation** - Accepts PDF, PNG, JPG, JPEG via CLI argument
2. **OCR extraction** - Tesseract with Dutch language pack (`lang='nld'`)
3. **Translation** - Google Gemini API (`gemini-2.0-flash` by default)
4. **Output** - Saves `{filename}_translated.txt` alongside original

Key dependencies:
- `pytesseract` - Python wrapper for Tesseract OCR
- `pdf2image` - Converts PDF pages to images (requires poppler-utils)
- `google-genai` - Google Gemini SDK

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `GOOGLE_API_KEY` | No | Alternative name for API key |

Get your API key at: https://aistudio.google.com/app/apikey

## Docker Configuration

- Base: `python:3.11-slim`
- System deps: `tesseract-ocr`, `tesseract-ocr-nld`, `poppler-utils`
- Runs as non-root user (`appuser`)
- Requires `GEMINI_API_KEY` environment variable
