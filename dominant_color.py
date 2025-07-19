"""
Lightweight Color Analysis Module for Fruit Ripeness Detection
=============================================================
This module uses 'colorgram.py' to provide efficient color analysis
without heavy dependencies like OpenCV or Scikit-learn, making it
ideal for resource-constrained environments.

Author: AI Assistant (Optimized for Render Free Tier)
Version: 1.0.0
"""
import colorgram
from PIL import Image, ImageDraw, ImageFont
import os
import uuid
import logging
from enum import Enum  # <== THÊM DÒNG NÀY
from dataclasses import dataclass # <== VÀ DÒNG NÀY
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DATA STRUCTURES ====================
class RipenessState(Enum):
    """Enumeration for ripeness states"""
    RIPE = "CHÍN"
    UNRIPE = "XANH"
    UNKNOWN = "KHÔNG XÁC ĐỊNH"

@dataclass
class ColorAnalysisResult:
    """Data class for complete color analysis result"""
    dominant_colors: list
    ripeness_state: RipenessState
    confidence: float
    visualization_path: str

# ==================== LIGHTWEIGHT COLOR ANALYZER ====================
class AdvancedColorAnalyzer:
    """A lightweight color analyzer using colorgram.py"""

    def __init__(self):
        self.images_dir = os.path.join(os.path.dirname(__file__), 'static', 'images')
        os.makedirs(self.images_dir, exist_ok=True)

    def analyze_image(self, image_path: str, bounding_box: tuple = None) -> ColorAnalysisResult:
        try:
            logger.info(f"Starting lightweight color analysis for: {image_path}")

            # Crop image if bounding box is provided
            if bounding_box:
                img = Image.open(image_path)
                x, y, w, h = bounding_box
                img = img.crop((x, y, x + w, y + h))
                analysis_target = img
            else:
                analysis_target = image_path

            # Extract 6 dominant colors
            colors = colorgram.extract(analysis_target, 6)
            
            green_pixels = 0
            yellow_orange_pixels = 0
            
            if not colors:
                raise ValueError("Could not extract any colors from the image.")

            for color in colors:
                r, g, b = color.rgb
                # Simple heuristic for green and yellow/orange
                if g > r and g > b + 10: # More green than other channels
                    green_pixels += color.proportion
                elif r > g and g > b: # Classic yellow/orange
                    yellow_orange_pixels += color.proportion

            total_significant_pixels = green_pixels + yellow_orange_pixels
            if total_significant_pixels == 0:
                ripeness = RipenessState.UNKNOWN
                confidence = 30.0
            else:
                green_ratio = green_pixels / total_significant_pixels
                
                if green_ratio > 0.5:
                    ripeness = RipenessState.UNRIPE
                    confidence = green_ratio * 100
                else:
                    ripeness = RipenessState.RIPE
                    confidence = (1 - green_ratio) * 100
            
            vis_path = self.create_visualization(image_path, colors, ripeness, confidence)

            return ColorAnalysisResult(
                dominant_colors=[{'rgb': c.rgb, 'proportion': c.proportion} for c in colors],
                ripeness_state=ripeness,
                confidence=round(confidence, 2),
                visualization_path=vis_path
            )

        except Exception as e:
            logger.error(f"Error during color analysis for {image_path}: {e}")
            return ColorAnalysisResult([], RipenessState.UNKNOWN, 0.0, "")

    def create_visualization(self, image_path: str, colors: list, ripeness: RipenessState, confidence: float) -> str:
        try:
            base_img = Image.open(image_path).convert("RGB").resize((224, 224))
            palette_height = 80
            vis_img = Image.new('RGB', (base_img.width, base_img.height + palette_height), (255, 255, 255))
            vis_img.paste(base_img, (0, 0))
            draw = ImageDraw.Draw(vis_img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()

            # Display ripeness text
            ripeness_text = f"Color Analysis: {ripeness.value} ({confidence:.1f}%)"
            draw.text((10, base_img.height + 5), ripeness_text, fill="black", font=font)
            
            # Draw color swatches
            bar_start_y = base_img.height + 30
            current_x = 0
            for color in colors:
                bar_width = int(color.proportion * base_img.width)
                draw.rectangle([current_x, bar_start_y, current_x + bar_width, vis_img.height], fill=color.rgb)
                current_x += bar_width

            filename = f"{uuid.uuid4()}_color_analysis.jpg"
            save_path = os.path.join(self.images_dir, filename)
            vis_img.save(save_path)
            
            return filename
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            return ""
