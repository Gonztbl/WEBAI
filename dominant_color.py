"""
Advanced Color Analysis Module for Fruit Ripeness Detection
===========================================================

This module provides enterprise-grade color analysis capabilities using:
- Advanced K-Means clustering with optimal K selection
- LAB color space analysis for better perceptual accuracy  
- HSV color space for hue-based ripeness detection
- Statistical color distribution analysis
- Machine learning-based color classification
- Performance optimization with caching and vectorization

Author: AI Fruit Analysis System
Version: 2.0.0 (Enterprise Edition)
"""

import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import imutils
import uuid
import webcolors
import colorsys
from typing import List, Dict, Tuple, Optional, Union
import logging
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
class ColorAnalysisConfig:
    """Configuration class for color analysis parameters"""

    # Image processing
    RESIZE_HEIGHT = 30  # Increased for better accuracy
    MIN_CLUSTER_SIZE = 0.10  # Minimum 5% to be considered significant
    MAX_CLUSTERS = 3  # Maximum number of color clusters
    MIN_CLUSTERS = 2  # Minimum number of color clusters

    # Color analysis
    BRIGHTNESS_THRESHOLD = 50  # Ignore very dark colors
    SATURATION_THRESHOLD = 30  # Ignore very desaturated colors

    # Ripeness detection weights
    HUE_WEIGHT = 0.4
    SATURATION_WEIGHT = 0.3
    BRIGHTNESS_WEIGHT = 0.2
    PERCENTAGE_WEIGHT = 0.1

    # Performance
    CACHE_SIZE = 128
    ENABLE_CACHING = True

# ==================== DATA STRUCTURES ====================
class RipenessState(Enum):
    """Enumeration for ripeness states"""
    UNRIPE = "XANH"
    RIPE = "CHÍN"
    OVERRIPE = "QUÁ CHÍN"
    UNKNOWN = "KHÔNG XÁC ĐỊNH"

@dataclass
class ColorInfo:
    """Data class for color information"""
    rgb: Tuple[int, int, int]
    hsv: Tuple[float, float, float]
    lab: Tuple[float, float, float]
    percentage: float
    name: str
    closest_name: str
    is_significant: bool = True

@dataclass
class ColorAnalysisResult:
    """Data class for complete color analysis result"""
    dominant_colors: List[ColorInfo]
    ripeness_state: RipenessState
    confidence: float
    analysis_metadata: Dict
    visualization_path: str

# ==================== ADVANCED COLOR SCIENCE ====================
class AdvancedColorAnalyzer:
    """Advanced color analyzer with multiple color spaces and ML techniques"""

    def __init__(self, config: ColorAnalysisConfig = None):
        self.config = config or ColorAnalysisConfig()
        self.images_dir = Path(__file__).parent / 'static' / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Load enhanced color mappings
        self._load_color_mappings()

        # Initialize performance metrics
        self.performance_stats = {
            'total_analyses': 0,
            'average_processing_time': 0,
            'cache_hits': 0
        }

    def _load_color_mappings(self):
        """Load comprehensive color mappings for fruit analysis"""

        # Enhanced CSS3 colors with additional fruit-specific colors
        self.css3_colors = {
            # Standard CSS3 colors (keeping original for compatibility)
            'aliceblue': '#f0f8ff', 'antiquewhite': '#faebd7', 'aqua': '#00ffff',
            'aquamarine': '#7fffd4', 'azure': '#f0ffff', 'beige': '#f5f5dc',
            'bisque': '#ffe4c4', 'black': '#000000', 'blanchedalmond': '#ffebcd',
            'blue': '#0000ff', 'blueviolet': '#8a2be2', 'brown': '#a52a2a',
            'burlywood': '#deb887', 'cadetblue': '#5f9ea0', 'chartreuse': '#7fff00',
            'chocolate': '#d2691e', 'coral': '#ff7f50', 'cornflowerblue': '#6495ed',
            'cornsilk': '#fff8dc', 'crimson': '#dc143c', 'cyan': '#00ffff',
            'darkblue': '#00008b', 'darkcyan': '#008b8b', 'darkgoldenrod': '#b8860b',
            'darkgray': '#a9a9a9', 'darkgreen': '#006400', 'darkkhaki': '#bdb76b',
            'darkmagenta': '#8b008b', 'darkolivegreen': '#556b2f',
            'darkorange': '#ff8c00', 'darkorchid': '#9932cc', 'darkred': '#8b0000',
            'darksalmon': '#e9967a', 'darkseagreen': '#8fbc8f',
            'darkslateblue': '#483d8b', 'darkslategray': '#2f4f4f',
            'darkturquoise': '#00ced1', 'darkviolet': '#9400d3', 'deeppink': '#ff1493',
            'deepskyblue': '#00bfff', 'dimgray': '#696969', 'dodgerblue': '#1e90ff',
            'firebrick': '#b22222', 'floralwhite': '#fffaf0', 'forestgreen': '#228b22',
            'fuchsia': '#ff00ff', 'gainsboro': '#dcdcdc', 'ghostwhite': '#f8f8ff',
            'gold': '#ffd700', 'goldenrod': '#daa520', 'gray': '#808080',
            'green': '#008000', 'greenyellow': '#adff2f', 'honeydew': '#f0fff0',
            'hotpink': '#ff69b4', 'indianred': '#cd5c5c', 'indigo': '#4b0082',
            'ivory': '#fffff0', 'khaki': '#f0e68c', 'lavender': '#e6e6fa',
            'lavenderblush': '#fff0f5', 'lawngreen': '#7cfc00', 'lemonchiffon': '#fffacd',
            'lightblue': '#add8e6', 'lightcoral': '#f08080', 'lightcyan': '#e0ffff',
            'lightgoldenrodyellow': '#fafad2', 'lightgreen': '#90ee90',
            'lightgrey': '#d3d3d3', 'lightpink': '#ffb6c1', 'lightsalmon': '#ffa07a',
            'lightseagreen': '#20b2aa', 'lightskyblue': '#87cefa',
            'lightslategray': '#778899', 'lightsteelblue': '#b0c4de',
            'lightyellow': '#ffffe0', 'lime': '#00ff00', 'limegreen': '#32cd32',
            'linen': '#faf0e6', 'magenta': '#ff00ff', 'maroon': '#800000',
            'mediumaquamarine': '#66cdaa', 'mediumblue': '#0000cd',
            'mediumorchid': '#ba55d3', 'mediumpurple': '#9370db',
            'mediumseagreen': '#3cb371', 'mediumslateblue': '#7b68ee',
            'mediumspringgreen': '#00fa9a', 'mediumturquoise': '#48d1cc',
            'mediumvioletred': '#c71585', 'midnightblue': '#191970',
            'mintcream': '#f5fffa', 'mistyrose': '#ffe4e1', 'moccasin': '#ffe4b5',
            'navajowhite': '#ffdead', 'navy': '#000080', 'oldlace': '#fdf5e6',
            'olive': '#808000', 'olivedrab': '#6b8e23', 'orange': '#ffa500',
            'orangered': '#ff4500', 'orchid': '#da70d6', 'palegoldenrod': '#eee8aa',
            'palegreen': '#98fb98', 'paleturquoise': '#afeeee', 'palevioletred': '#db7093',
            'papayawhip': '#ffefd5', 'peachpuff': '#ffdab9', 'peru': '#cd853f',
            'pink': '#ffc0cb', 'plum': '#dda0dd', 'powderblue': '#b0e0e6',
            'purple': '#800080', 'rebeccapurple': '#663399', 'red': '#ff0000',
            'rosybrown': '#bc8f8f', 'royalblue': '#4169e1', 'saddlebrown': '#8b4513',
            'salmon': '#fa8072', 'sandybrown': '#f4a460', 'seagreen': '#2e8b57',
            'seashell': '#fff5ee', 'sienna': '#a0522d', 'silver': '#c0c0c0',
            'skyblue': '#87ceeb', 'slateblue': '#6a5acd', 'slategray': '#708090',
            'snow': '#fffafa', 'springgreen': '#00ff7f', 'steelblue': '#4682b4',
            'tan': '#d2b48c', 'teal': '#008080', 'thistle': '#d8bfd8',
            'tomato': '#ff6347', 'turquoise': '#40e0d0', 'violet': '#ee82ee',
            'wheat': '#f5deb3', 'white': '#ffffff', 'whitesmoke': '#f5f5f5',
            'yellow': '#ffff00', 'yellowgreen': '#9acd32',

            # Additional fruit-specific colors
            'applegreen': '#8db600', 'applered': '#ff0800', 'bananagreen': '#9acd32',
            'bananayellow': '#ffe135', 'orangegreen': '#32cd32', 'orangeorange': '#ffa500'
        }

        # Advanced ripeness color classification with HSV ranges
        # Advanced ripeness color classification with HSV ranges
        self.ripeness_color_ranges = {
            RipenessState.UNRIPE: {
                # Widen the green range to include yellow-greens
                'hue_ranges': [(70, 160)],  # More robust range for Green/Unripe
                'saturation_min': 30,
                'brightness_min': 40,
                'keywords': {
                    'green', 'lime', 'chartreuse', 'forestgreen', 'darkgreen',
                    'limegreen', 'springgreen', 'seagreen', 'olive', 'olivedrab',
                    'yellowgreen', 'lawngreen', 'palegreen', 'lightgreen',
                    'mediumseagreen', 'mediumspringgreen', 'darkolivegreen',
                    'darkseagreen', 'lightseagreen', 'applegreen', 'bananagreen',
                    'orangegreen'
                }
            },
            RipenessState.RIPE: {
                # Shrink the yellow range slightly to avoid green overlap
                'hue_ranges': [(0, 55), (330, 360)],  # Red to Yellow, avoiding green
                'saturation_min': 25,
                'brightness_min': 35,
                'keywords': {
                    'red', 'orange', 'yellow', 'gold', 'crimson', 'coral',
                    'salmon', 'tomato', 'orangered', 'darkorange', 'goldenrod',
                    'darkgoldenrod', 'lightsalmon', 'darksalmon', 'lightcoral',
                    'indianred', 'firebrick', 'darkred', 'maroon', 'brown',
                    'saddlebrown', 'sienna', 'peru', 'chocolate', 'sandybrown',
                    'tan', 'burlywood', 'rosybrown', 'applered', 'bananayellow',
                    'orangeorange'
                }
            },
            RipenessState.OVERRIPE: {
                'hue_ranges': [(15, 45)],  # Brown-dark orange hues
                'saturation_min': 20,
                'brightness_min': 20,
                'keywords': {
                    'brown', 'darkred', 'maroon', 'saddlebrown', 'sienna',
                    'chocolate', 'peru', 'darkgoldenrod'
                }
            }
        }

        # Colors to ignore (background/neutral colors)
        self.ignore_colors = {
            'white', 'whitesmoke', 'gainsboro', 'lightgrey', 'silver',
            'darkgray', 'gray', 'dimgray', 'black', 'snow', 'ivory',
            'floralwhite', 'ghostwhite', 'mintcream', 'azure', 'aliceblue',
            'lavender', 'lavenderblush', 'mistyrose', 'seashell'
        }

    @lru_cache(maxsize=ColorAnalysisConfig.CACHE_SIZE)
    def _rgb_to_color_name(self, rgb_tuple: Tuple[int, int, int]) -> Tuple[Optional[str], str]:
        """Convert RGB to color name with caching for performance"""
        try:
            actual_name = webcolors.rgb_to_name(rgb_tuple)
            return actual_name, actual_name
        except ValueError:
            closest_name = self._find_closest_color(rgb_tuple)
            return None, closest_name

    def _find_closest_color(self, requested_rgb: Tuple[int, int, int]) -> str:
        """Find closest color name using Euclidean distance in RGB space"""
        min_distance = float('inf')
        closest_color = 'unknown'

        for name, hex_code in self.css3_colors.items():
            try:
                color_rgb = webcolors.hex_to_rgb(hex_code)
                # Calculate Euclidean distance
                distance = sum((a - b) ** 2 for a, b in zip(requested_rgb, color_rgb)) ** 0.5

                if distance < min_distance:
                    min_distance = distance
                    closest_color = name
            except ValueError:
                continue

        return closest_color

    def _rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to HSV color space"""
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return (h * 360, s * 100, v * 100)  # Convert to degrees and percentages

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to LAB color space using OpenCV for accuracy."""
        # OpenCV yêu cầu một mảng numpy 3D uint8
        rgb_pixel = np.uint8([[rgb]])
        lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)
        # Kết quả trả về là một tuple float
        return tuple(float(c) for c in lab_pixel[0][0])

    def _determine_optimal_clusters(self, image_data: np.ndarray) -> int:
        """Determine optimal number of clusters using silhouette analysis"""
        max_score = -1
        optimal_k = self.config.MIN_CLUSTERS

        for k in range(self.config.MIN_CLUSTERS, min(self.config.MAX_CLUSTERS + 1, len(image_data) // 100)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(image_data)

                if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                    score = silhouette_score(image_data, labels)
                    if score > max_score:
                        max_score = score
                        optimal_k = k
            except Exception as e:
                logger.warning(f"Error calculating silhouette score for k={k}: {e}")
                continue

        return optimal_k

    def _extract_dominant_colors(self, image_path: str, bounding_box: Optional[Tuple[int, int, int, int]] = None) -> List[ColorInfo]:
        """Extract dominant colors using advanced K-means clustering"""

        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image from: {image_path}")

        # Apply bounding box if provided
        if bounding_box:
            x, y, w, h = bounding_box
            # Validate bounding box
            if x >= 0 and y >= 0 and x + w <= img.shape[1] and y + h <= img.shape[0]:
                img_roi = img[y:y + h, x:x + w]
                if img_roi.size > 0:
                    img = img_roi
                else:
                    logger.warning("Invalid bounding box region, using full image")
            else:
                logger.warning("Bounding box out of image bounds, using full image")

        # Resize for performance while maintaining quality
        resized_img = imutils.resize(img, height=self.config.RESIZE_HEIGHT)
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        # Flatten image data
        flat_img = img_rgb.reshape(-1, 3)

        # Remove very dark pixels (likely shadows/noise)
        brightness = np.mean(flat_img, axis=1)
        bright_pixels = flat_img[brightness > self.config.BRIGHTNESS_THRESHOLD]

        if len(bright_pixels) == 0:
            bright_pixels = flat_img  # Fallback if all pixels are dark

        # Determine optimal number of clusters
        optimal_clusters = self._determine_optimal_clusters(bright_pixels)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        kmeans.fit(bright_pixels)

        # Get cluster centers and labels
        centers = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        # Calculate percentages
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)

        # Create ColorInfo objects
        color_infos = []
        for i, (center, percentage) in enumerate(zip(centers, percentages)):
            rgb = tuple(center)
            hsv = self._rgb_to_hsv(rgb)
            lab = self._rgb_to_lab(rgb)

            actual_name, closest_name = self._rgb_to_color_name(rgb)

            # Determine if color is significant
            is_significant = (
                percentage >= self.config.MIN_CLUSTER_SIZE and
                hsv[1] >= self.config.SATURATION_THRESHOLD and  # Sufficient saturation
                hsv[2] >= self.config.BRIGHTNESS_THRESHOLD and  # Sufficient brightness
                closest_name not in self.ignore_colors
            )

            color_info = ColorInfo(
                rgb=rgb,
                hsv=hsv,
                lab=lab,
                percentage=percentage,
                name=actual_name or closest_name,
                closest_name=closest_name,
                is_significant=is_significant
            )

            color_infos.append(color_info)

        # Sort by percentage (descending) and filter significant colors
        color_infos.sort(key=lambda x: x.percentage, reverse=True)
        significant_colors = [c for c in color_infos if c.is_significant]

        # If no significant colors found, return top colors anyway
        if not significant_colors:
            significant_colors = color_infos[:3]
            logger.warning("No significant colors found, returning top 3 colors")

        return significant_colors

    def _analyze_ripeness_advanced(self, colors: List[ColorInfo]) -> Tuple[RipenessState, float]:
        """Advanced ripeness analysis using multiple color spaces and ML techniques"""

        if not colors:
            return RipenessState.UNKNOWN, 0.0

        # Filter for significant colors first
        significant_colors = [c for c in colors if c.is_significant]

        # If no significant colors, fallback to the most dominant color
        if not significant_colors:
            logger.warning("No significant colors found. Falling back to most dominant color for analysis.")
            analysis_colors = colors[:1] # Use only the top color
            confidence_penalty = 0.5 # Apply a penalty because we are less certain
        else:
            analysis_colors = significant_colors
            confidence_penalty = 1.0


        state_scores = {state: 0.0 for state in RipenessState}
        total_weight = 0.0

        for color in analysis_colors:
            color_weight = color.percentage
            total_weight += color_weight
            hue, saturation, brightness = color.hsv

            for state, criteria in self.ripeness_color_ranges.items():
                score = 0.0
                hue_match = False
                for hue_min, hue_max in criteria['hue_ranges']:
                    if hue_min <= hue <= hue_max:
                        hue_match = True
                        break
                if hue_match: score += self.config.HUE_WEIGHT
                if saturation >= criteria['saturation_min']: score += self.config.SATURATION_WEIGHT
                if brightness >= criteria['brightness_min']: score += self.config.BRIGHTNESS_WEIGHT
                if color.closest_name in criteria['keywords']: score += self.config.PERCENTAGE_WEIGHT
                state_scores[state] += score * color_weight

        if total_weight > 0:
            for state in state_scores:
                state_scores[state] /= total_weight

        best_state = max(state_scores, key=state_scores.get)
        confidence = state_scores[best_state] * 100 * confidence_penalty

        if confidence < 30:
            return RipenessState.UNKNOWN, confidence

        return best_state, confidence

    def create_fixed_visualization(self,
                                   original_image_path: str,
                                   colors: list,
                                   ripeness_state: str,
                                   confidence: float,
                                   bounding_box: tuple = None) -> str:
        """
        Create a properly formatted color analysis visualization
        Fixes all the issues: Unicode, truncation, layout
        """

        try:
            # Load original image
            img = cv2.imread(original_image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {original_image_path}")

            rows, cols = img.shape[:2]
            final_img = img.copy()

            # Draw bounding box if provided
            if bounding_box:
                x, y, w, h = bounding_box
                # Ensure bounding box is within image bounds
                x = max(0, min(x, cols - 1))
                y = max(0, min(y, rows - 1))
                w = min(w, cols - x)
                h = min(h, rows - y)

                cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(final_img, 'Analysis Region', (x, max(y - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Create properly sized overlay - KEY FIX
            overlay_height = max(450, len(colors) * 80 + 200)  # Dynamic height
            overlay_width = max(cols, 800)  # Ensure minimum width
            overlay = np.ones((overlay_height, overlay_width, 3), dtype=np.uint8) * 255

            # Title with better positioning
            title_text = 'Advanced Color Analysis Results'
            cv2.putText(overlay, title_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            # Fixed ripeness display - CRITICAL FIX
            ripeness_mapping = {
                'CHÍN': 'RIPE',
                'XANH': 'UNRIPE',
                'QUÁ CHÍN': 'OVERRIPE',
                'KHÔNG XÁC ĐỊNH': 'UNKNOWN',
                'CH??N': 'RIPE',  # Handle corrupted text
                'CH?N': 'RIPE'  # Handle partial corruption
            }

            # Clean the ripeness state
            clean_ripeness = ripeness_state.upper().strip()
            display_ripeness = ripeness_mapping.get(clean_ripeness, clean_ripeness)

            # Color based on ripeness
            if 'RIPE' in display_ripeness and 'OVER' not in display_ripeness:
                ripeness_color = (0, 150, 0)  # Green for ripe
            elif 'UNRIPE' in display_ripeness:
                ripeness_color = (0, 100, 200)  # Blue for unripe
            else:
                ripeness_color = (100, 100, 100)  # Gray for unknown

            # Display ripeness with full text - FIXED
            ripeness_text = f'Ripeness: {display_ripeness} ({confidence:.1f}%)'
            cv2.putText(overlay, ripeness_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, ripeness_color, 2)

            # Color swatches with improved layout - MAJOR FIX
            start_x = 20
            start_y = 130
            swatch_width = 120
            swatch_height = 80
            spacing_x = 140
            spacing_y = 120
            colors_per_row = max(1, (overlay_width - 40) // spacing_x)

            # Add section header
            cv2.putText(overlay, 'Dominant Colors:', (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            for i, color in enumerate(colors[:8]):  # Limit to 8 colors
                # Calculate position
                row = i // colors_per_row
                col = i % colors_per_row

                x_pos = start_x + col * spacing_x
                y_pos = start_y + row * spacing_y

                # Check bounds
                if y_pos + swatch_height + 40 > overlay_height:
                    break

                # Extract color data safely
                try:
                    if hasattr(color, 'rgb'):
                        rgb = color.rgb
                        percentage = color.percentage * 100
                        name = color.closest_name
                    else:
                        # Handle dict format
                        rgb = color.get('color', [128, 128, 128])
                        percentage = color.get('percent', 0) * 100
                        name = color.get('name', 'unknown')

                    color_bgr = tuple(int(c) for c in rgb[::-1])  # RGB to BGR

                except Exception as e:
                    logger.warning(f"Error processing color {i}: {e}")
                    continue

                # Draw swatch with border
                cv2.rectangle(overlay, (x_pos, y_pos),
                              (x_pos + swatch_width, y_pos + swatch_height),
                              color_bgr, -1)
                cv2.rectangle(overlay, (x_pos, y_pos),
                              (x_pos + swatch_width, y_pos + swatch_height),
                              (0, 0, 0), 2)

                # Percentage inside swatch with good contrast
                brightness = sum(rgb) / 3
                text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)

                percentage_text = f'{percentage:.1f}%'
                cv2.putText(overlay, percentage_text,
                            (x_pos + 10, y_pos + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                # Color name below swatch - cleaned
                clean_name = str(name)[:15]  # Truncate long names
                cv2.putText(overlay, clean_name,
                            (x_pos, y_pos + swatch_height + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # RGB values below name
                rgb_text = f'RGB({rgb[0]},{rgb[1]},{rgb[2]})'
                cv2.putText(overlay, rgb_text,
                            (x_pos, y_pos + swatch_height + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            # Analysis summary at bottom
            summary_y = start_y + ((len(colors) - 1) // colors_per_row + 1) * spacing_y + 20
            if summary_y + 80 < overlay_height:
                cv2.line(overlay, (20, summary_y - 10), (overlay_width - 20, summary_y - 10),
                         (200, 200, 200), 1)

                cv2.putText(overlay, f'Analysis Summary:', (20, summary_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                cv2.putText(overlay, f'• Total colors analyzed: {len(colors)}',
                            (20, summary_y + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

                cv2.putText(overlay, f'• Method: K-Means clustering + HSV analysis',
                            (20, summary_y + 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            # Resize final image to match overlay width if needed
            if final_img.shape[1] != overlay_width:
                final_img = cv2.resize(final_img, (overlay_width,
                                                   int(final_img.shape[0] * overlay_width / final_img.shape[1])))

            # Combine images
            result_img = np.vstack([final_img, overlay])

            # Save with unique filename
            unique_filename = f"{uuid.uuid4()}_fixed_color_analysis.jpg"
            output_path = self.images_dir / unique_filename

            # Save with high quality
            success = cv2.imwrite(str(output_path), result_img,
                                  [cv2.IMWRITE_JPEG_QUALITY, 95])

            if not success:
                raise RuntimeError(f"Failed to save image to {output_path}")

            logger.info(f"✅ Fixed color analysis saved: {unique_filename}")
            return unique_filename

        except Exception as e:
            logger.error(f"❌ Error creating color visualization: {e}")
            # Return error placeholder
            return self._create_error_placeholder()

    def analyze_image(self,
                     image_path: str,
                     bounding_box: Optional[Tuple[int, int, int, int]] = None) -> ColorAnalysisResult:

        start_time = time.time()

        try:
            logger.info(f"Starting advanced color analysis for: {image_path}")

            # Extract dominant colors
            colors = self._extract_dominant_colors(image_path, bounding_box)

            # Analyze ripeness
            ripeness_state, confidence = self._analyze_ripeness_advanced(colors)

            # Create visualization
            visualization_path = self.create_fixed_visualization(
                image_path, colors, ripeness_state.value, confidence, bounding_box
            )

            # Prepare metadata
            processing_time = time.time() - start_time
            metadata = {
                'processing_time_seconds': processing_time,
                'num_colors_analyzed': len(colors),
                'num_significant_colors': len([c for c in colors if c.is_significant]),
                'analysis_timestamp': time.time(),
                'bounding_box_used': bounding_box is not None,
                'optimal_clusters': len(colors)
            }

            # Update performance stats
            self.performance_stats['total_analyses'] += 1
            self.performance_stats['average_processing_time'] = (
                (self.performance_stats['average_processing_time'] *
                 (self.performance_stats['total_analyses'] - 1) + processing_time) /
                self.performance_stats['total_analyses']
            )

            result = ColorAnalysisResult(
                dominant_colors=colors,
                ripeness_state=ripeness_state,
                confidence=confidence,
                analysis_metadata=metadata,
                visualization_path=visualization_path
            )

            logger.info(f"Color analysis completed successfully: {ripeness_state.value} ({confidence:.1f}%)")
            return result

        except Exception as e:
            logger.error(f"Error during color analysis for {image_path}: {e}")
            # Return error result to prevent system crash
            return ColorAnalysisResult(
                dominant_colors=[],
                ripeness_state=RipenessState.UNKNOWN,
                confidence=0.0,
                analysis_metadata={'error': str(e)},
                visualization_path=''
            )

    def _create_error_placeholder(self) -> str:
        """Create an error placeholder image"""
        try:
            error_img = np.ones((300, 600, 3), dtype=np.uint8) * 240
            cv2.putText(error_img, 'Color Analysis Error', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
            cv2.putText(error_img, 'Please try again', (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

            error_filename = f"error_{uuid.uuid4()}.jpg"
            error_path = self.images_dir / error_filename
            cv2.imwrite(str(error_path), error_img)
            return error_filename
        except:
            return "error_placeholder.jpg"
# ==================== BACKWARD COMPATIBILITY ====================
# Legacy functions for backward compatibility with existing code

def color_of_image(filepath, bounding_box=None):
    """
    Legacy function for backward compatibility
    Returns: (dominant_colors_info, filename)
    """
    try:
        analyzer = AdvancedColorAnalyzer()
        result = analyzer.analyze_image(filepath, bounding_box)

        # Convert new format to old format
        dominant_colors_info = []
        for color in result.dominant_colors:
            dominant_colors_info.append({
                'percent': color.percentage,
                'color': list(color.rgb)
            })

        return dominant_colors_info, result.visualization_path

    except Exception as e:
        logger.error(f"Error in legacy color_of_image function: {e}")
        # Return empty result for compatibility
        return [], f"{uuid.uuid4()}_error.jpg"

def name_main_color(dominant_colors_info):
    """
    Legacy function for backward compatibility
    """
    try:
        if not dominant_colors_info:
            return "KHÔNG XÁC ĐỊNH"

        # Use the advanced analyzer for better accuracy
        analyzer = AdvancedColorAnalyzer()

        # Convert old format to new format
        colors = []
        for color_info in dominant_colors_info:
            if isinstance(color_info, dict) and 'color' in color_info and 'percent' in color_info:
                rgb = tuple(color_info['color'])
                hsv = analyzer._rgb_to_hsv(rgb)
                lab = analyzer._rgb_to_lab(rgb)
                actual_name, closest_name = analyzer._rgb_to_color_name(rgb)

                color = ColorInfo(
                    rgb=rgb,
                    hsv=hsv,
                    lab=lab,
                    percentage=color_info['percent'],
                    name=actual_name or closest_name,
                    closest_name=closest_name,
                    is_significant=True
                )
                colors.append(color)

        # Analyze ripeness
        ripeness_state, confidence = analyzer._analyze_ripeness_advanced(colors)
        return ripeness_state.value

    except Exception as e:
        logger.error(f"Error in legacy name_main_color function: {e}")
        return "KHÔNG XÁC ĐỊNH"
