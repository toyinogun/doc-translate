#!/usr/bin/env python3
"""
PDF Builder - Create translated PDFs with preserved layout.

This module creates output PDFs with the original page as background
and translated text overlaid in the same positions.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image


@dataclass
class TextBlock:
    """Represents a text block with position and content."""
    text: str
    translated_text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float = 0.0


class TranslatedPDFBuilder:
    """
    Builds translated PDFs with original pages as backgrounds.

    Uses PyMuPDF to create PDFs where:
    - Original page images are set as backgrounds
    - White rectangles cover original text positions
    - Translated text is inserted in the same positions
    """

    # Minimum font size for readability
    MIN_FONT_SIZE = 5
    # Maximum font size
    MAX_FONT_SIZE = 14
    # Font to use for translated text
    FONT_NAME = "helv"  # Helvetica

    def __init__(self):
        """Initialize the PDF builder."""
        self.doc = fitz.open()

    def add_page_with_translation(
        self,
        background_image: Image.Image,
        text_blocks: list[TextBlock],
        dpi: int = 200,
        use_background: bool = False
    ) -> None:
        """
        Add a page with translated text, optionally with original image as background.

        Args:
            background_image: Original page image (used for dimensions, optionally as background)
            text_blocks: List of TextBlock objects with translations
            dpi: DPI used when creating the image (for coordinate conversion)
            use_background: If True, use original image as background; if False, white background
        """
        # Calculate page dimensions in points (72 points per inch)
        # PDF coordinates are in points, image coordinates are in pixels at given DPI
        scale = 72.0 / dpi
        page_width = background_image.width * scale
        page_height = background_image.height * scale

        # Create a new page with the correct dimensions
        page = self.doc.new_page(width=page_width, height=page_height)

        # Optionally insert the background image
        if use_background:
            img_bytes = io.BytesIO()
            if background_image.mode != 'RGB':
                background_image = background_image.convert('RGB')
            background_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            page_rect = fitz.Rect(0, 0, page_width, page_height)
            page.insert_image(page_rect, stream=img_bytes.getvalue())

        # Insert translated text blocks
        for block in text_blocks:
            if block.translated_text.strip():
                self._insert_text_block(page, block, scale, page_width, draw_white_rect=use_background)

    def _insert_text_block(
        self,
        page: fitz.Page,
        block: TextBlock,
        scale: float,
        page_width: float,
        draw_white_rect: bool = False
    ) -> None:
        """
        Insert a translated text block onto the page.

        Args:
            page: PyMuPDF page object
            block: TextBlock with translation
            scale: Scale factor from pixels to points
            page_width: Page width in points
            draw_white_rect: If True, draw white rectangle to cover original text
        """
        # Convert pixel coordinates to PDF points
        x0 = block.left * scale
        y0 = block.top * scale
        x1 = (block.left + block.width) * scale
        y1 = (block.top + block.height) * scale

        box_width = x1 - x0
        box_height = y1 - y0

        # Calculate font size to fit text in original box width
        fontsize = self._calculate_fit_fontsize(
            block.translated_text,
            box_width,
            box_height
        )

        # Optionally draw white rectangle to cover original text
        if draw_white_rect:
            left_start = min(x0, 15)
            v_padding = 2
            h_padding = 5
            rect = fitz.Rect(left_start, y0 - v_padding, x1 + h_padding, y1 + v_padding)
            shape = page.new_shape()
            shape.draw_rect(rect)
            shape.finish(color=(1, 1, 1), fill=(1, 1, 1))
            shape.commit()

        # Insert translated text at original x position
        baseline_y = y0 + box_height * 0.75

        page.insert_text(
            fitz.Point(x0, baseline_y),
            block.translated_text,
            fontsize=fontsize,
            fontname=self.FONT_NAME,
            color=(0, 0, 0),
        )

    def _calculate_fit_fontsize(
        self,
        text: str,
        box_width: float,
        box_height: float
    ) -> float:
        """
        Calculate font size that fits text within the given box dimensions.

        Considers both box height and width to ensure text fits.

        Args:
            text: Text to fit
            box_width: Available width in points
            box_height: Available height in points

        Returns:
            Font size in points (between MIN_FONT_SIZE and MAX_FONT_SIZE)
        """
        if not text.strip():
            return self.MIN_FONT_SIZE

        # Start with font size based on box height (80% of height)
        height_based_size = box_height * 0.8

        # Calculate font size based on width
        # Average character width in Helvetica is ~0.5 * fontsize
        # So: text_len * 0.5 * fontsize <= box_width
        # fontsize <= box_width / (text_len * 0.5)
        text_len = len(text)
        if text_len > 0:
            width_based_size = box_width / (text_len * 0.52)
        else:
            width_based_size = height_based_size

        # Use the smaller of the two to ensure text fits
        fontsize = min(height_based_size, width_based_size)

        # Clamp to reasonable bounds
        fontsize = max(self.MIN_FONT_SIZE, min(fontsize, self.MAX_FONT_SIZE))

        return fontsize

    def save(self, output_path: Path | str) -> Path:
        """
        Save the PDF document to a file.

        Args:
            output_path: Path where the PDF should be saved

        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)
        self.doc.save(str(output_path))
        return output_path

    def close(self) -> None:
        """Close the PDF document and release resources."""
        self.doc.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures document is closed."""
        self.close()
        return False
