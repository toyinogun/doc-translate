#!/usr/bin/env python3
"""
DocuTranslate Web UI - Gradio Interface

A simple web interface for uploading and translating Dutch documents.
"""

import os
import tempfile
from pathlib import Path

import gradio as gr

# Import core functions from main.py
from main import (
    SUPPORTED_EXTENSIONS,
    DEFAULT_MODEL,
    extract_text_from_pdf,
    extract_text_from_image_file,
    translate_text,
    process_pdf_with_layout,
)


def check_api_key() -> tuple[bool, str]:
    """Check if Gemini API key is configured."""
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if api_key:
        return True, "API key configured"
    return False, "GEMINI_API_KEY not set. Please set it in your environment."


def process_document(
    file,
    model: str,
    output_format: str = "text",
    progress=gr.Progress()
) -> tuple[str, str, str, str | None]:
    """
    Process uploaded document: OCR and translate.

    Args:
        file: Uploaded file object
        model: Gemini model to use
        output_format: "text" for plain text or "pdf" for layout-preserved PDF
        progress: Gradio progress tracker

    Returns:
        Tuple of (original_text, translated_text, status_message, pdf_path)
    """
    # Check API key first
    has_key, key_status = check_api_key()
    if not has_key:
        return "", "", f"Error: {key_status}", None

    if file is None:
        return "", "", "Please upload a file first.", None

    # Get file path and extension
    file_path = Path(file.name)
    extension = file_path.suffix.lower()

    # Validate file type
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return "", "", f"Unsupported file type: {extension}. Supported: {supported}", None

    try:
        # PDF output with preserved layout
        if output_format == "pdf" and extension == '.pdf':
            def progress_callback(pct, msg):
                progress(pct, desc=msg)

            pdf_path = process_pdf_with_layout(file_path, model, progress_callback)

            return (
                "(Text extraction skipped for PDF output)",
                "(See downloaded PDF for translation)",
                "Translation complete! Download the PDF below.",
                str(pdf_path)
            )

        # Standard text output
        progress(0.1, desc="Extracting text with OCR...")

        if extension == '.pdf':
            original_text = extract_text_from_pdf(file_path)
        else:
            original_text = extract_text_from_image_file(file_path)

        if not original_text.strip():
            return "", "", "No text could be extracted. Please ensure the document contains readable text.", None

        progress(0.5, desc="Text extracted. Translating...")

        # Step 2: Translate
        translated_text = translate_text(original_text, model)

        progress(1.0, desc="Complete!")

        return original_text, translated_text, "Translation complete!", None

    except Exception as e:
        return "", "", f"Error: {str(e)}", None


def create_download_file(original: str, translated: str, filename: str) -> str | None:
    """Create a downloadable text file with the translation."""
    if not translated:
        return None

    content = f"""{'='*60}
DOCUTRANSLATE - Dutch to English Translation
{'='*60}
Source file: {filename}
{'='*60}

{'='*60}
ORIGINAL DUTCH TEXT
{'='*60}

{original}

{'='*60}
ENGLISH TRANSLATION
{'='*60}

{translated}
"""

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='_translated.txt', delete=False) as f:
        f.write(content)
        return f.name


# Build the Gradio interface
def create_ui():
    """Create and return the Gradio interface."""

    # Check API key status for display
    has_key, key_status = check_api_key()

    with gr.Blocks(title="DocuTranslate") as app:
        gr.Markdown(
            """
            # DocuTranslate
            ### Dutch to English Document Translation

            Upload a Dutch document (PDF or image) to extract text and translate it to English.
            """
        )

        # API key status
        if not has_key:
            gr.Markdown(
                """
                > **Warning:** GEMINI_API_KEY environment variable is not set.
                > Set it before uploading: `export GEMINI_API_KEY="your-key"`
                """,
                elem_classes=["warning"]
            )

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                file_input = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    type="filepath"
                )

                model_dropdown = gr.Dropdown(
                    choices=[
                        ("Gemini 2.0 Flash (Fast)", "gemini-2.0-flash"),
                        ("Gemini 2.5 Flash (Balanced)", "gemini-2.5-flash"),
                        ("Gemini 2.5 Pro (Best Quality)", "gemini-2.5-pro"),
                    ],
                    value="gemini-2.0-flash",
                    label="Translation Model"
                )

                output_format = gr.Radio(
                    choices=[
                        ("Plain Text", "text"),
                        ("Translated PDF (preserves layout)", "pdf"),
                    ],
                    value="text",
                    label="Output Format",
                    visible=False  # Only shown for PDF uploads
                )

                translate_btn = gr.Button("Translate", variant="primary", size="lg")

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready. Upload a document to begin."
                )

        with gr.Row():
            with gr.Column():
                original_text = gr.Textbox(
                    label="Original Dutch Text",
                    lines=15,
                    max_lines=30,
                    interactive=False
                )

            with gr.Column():
                translated_text = gr.Textbox(
                    label="English Translation",
                    lines=15,
                    max_lines=30,
                    interactive=False
                )

        with gr.Row():
            download_btn = gr.Button("Download Translation", variant="secondary")
            download_file = gr.File(label="Download", visible=False)

        # Store filename and PDF path for download
        filename_state = gr.State("")
        pdf_path_state = gr.State(None)

        # Event handlers
        def on_file_upload(file):
            if file:
                file_path = Path(file.name)
                # Show output format options only for PDF files
                is_pdf = file_path.suffix.lower() == '.pdf'
                return file_path.name, gr.Radio(visible=is_pdf)
            return "", gr.Radio(visible=False)

        file_input.change(
            fn=on_file_upload,
            inputs=[file_input],
            outputs=[filename_state, output_format]
        )

        def on_translate(file, model, out_format):
            original, translated, status, pdf_path = process_document(
                file, model, out_format
            )
            return original, translated, status, pdf_path

        translate_btn.click(
            fn=on_translate,
            inputs=[file_input, model_dropdown, output_format],
            outputs=[original_text, translated_text, status_text, pdf_path_state]
        )

        def on_download(original, translated, filename, pdf_path):
            # If we have a PDF path, return that
            if pdf_path:
                return gr.File(value=pdf_path, visible=True)
            # Otherwise create a text file
            if translated and translated != "(See downloaded PDF for translation)":
                filepath = create_download_file(original, translated, filename or "document")
                return gr.File(value=filepath, visible=True)
            return gr.File(visible=False)

        download_btn.click(
            fn=on_download,
            inputs=[original_text, translated_text, filename_state, pdf_path_state],
            outputs=[download_file]
        )

        gr.Markdown(
            """
            ---
            **Supported formats:** PDF, PNG, JPG, JPEG
            **Note:** Document text is sent to Google's Gemini API for translation.
            """
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft()
    )
