#!/usr/bin/env python3
"""Create a test image with Dutch text for testing DocuTranslate."""

from PIL import Image, ImageDraw, ImageFont

# Create a white image
img = Image.new('RGB', (800, 400), color='white')
draw = ImageDraw.Draw(img)

# Dutch text sample (a simple bank letter excerpt)
dutch_text = """Geachte heer/mevrouw,

Wij bevestigen de ontvangst van uw aanvraag.
Uw rekeningnummer is succesvol geactiveerd.
Het huidige saldo bedraagt vijfhonderd euro.

Met vriendelijke groet,
De Bank"""

# Use default font (will work without installing fonts)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
except:
    font = ImageFont.load_default()

# Draw the text
draw.text((50, 50), dutch_text, fill='black', font=font)

# Save the image
img.save('/documents/test_dutch.png')
print("Created: /documents/test_dutch.png")
