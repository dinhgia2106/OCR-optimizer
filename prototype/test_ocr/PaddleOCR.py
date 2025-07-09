import cv2
import json
import os
import sys
from pathlib import Path
import difflib
from typing import Dict, List, Tuple
import re
import time
import numpy as np

# Add the parent directory to the path to access test_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PaddleOCRTester:
    def __init__(self):
        # Get the correct path to test_data relative to this script
        current_dir = Path(__file__).parent
        self.test_data_path = current_dir.parent / "test_data"
        self.results = {}
        
        # Initialize PaddleOCR components
        self._init_paddleocr()
        
    def _init_paddleocr(self):
        """Initialize PaddleOCR components"""
        try:
            from paddleocr import PaddleOCR
            
            # Initialize PaddleOCR with English and French support
            print("Initializing PaddleOCR...")
            self.ocr_en = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.ocr_fr = PaddleOCR(use_angle_cls=True, lang='french', show_log=False)
            print("PaddleOCR initialized successfully")
            
        except ImportError as e:
            raise ImportError(f"PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr. Error: {e}")
        except Exception as e:
            print(f"Warning: Error initializing PaddleOCR: {e}")
            print("Trying with basic configuration...")
            try:
                self.ocr_en = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                self.ocr_fr = self.ocr_en  # Fallback to English for French
                print("PaddleOCR initialized with fallback configuration")
            except Exception as e2:
                raise ImportError(f"Could not initialize PaddleOCR: {e2}")
        
    def preprocess_image(self, image_path: str) -> any:
        """Preprocess image for better OCR results"""
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def extract_text_paddleocr(self, image_path: str, language: str = 'english') -> str:
        """Extract text using PaddleOCR"""
        try:
            # Choose appropriate OCR model
            ocr = self.ocr_en if language == 'english' else self.ocr_fr
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return ""
            
            # Run OCR
            result = ocr.ocr(img, cls=True)
            
            # Extract text from results
            text_parts = []
            if result and len(result) > 0:
                for line in result:
                    if line:  # Check if line is not None
                        for word_info in line:
                            if len(word_info) >= 2:
                                # word_info format: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
                                text = word_info[1][0] if isinstance(word_info[1], tuple) else word_info[1]
                                confidence = word_info[1][1] if isinstance(word_info[1], tuple) and len(word_info[1]) > 1 else 1.0
                                
                                # Filter by confidence threshold
                                if confidence > 0.1:
                                    text_parts.append(str(text))
            
            combined_text = ' '.join(text_parts)
            
            # Clean extracted text
            combined_text = combined_text.strip()
            combined_text = re.sub(r'\s+', ' ', combined_text)  # Replace multiple spaces with single space
            
            return combined_text
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return ""
    
    def calculate_similarity_metrics(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """Calculate various similarity metrics"""
        # Sequence Matcher ratio
        seq_matcher = difflib.SequenceMatcher(None, predicted, ground_truth)
        similarity_ratio = seq_matcher.ratio()
        
        # Calculate edit distance (Levenshtein)
        def edit_distance(s1, s2):
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        edit_dist = edit_distance(predicted, ground_truth)
        max_len = max(len(predicted), len(ground_truth))
        normalized_edit_distance = 1 - (edit_dist / max_len) if max_len > 0 else 1.0
        
        # Word-level accuracy
        predicted_words = predicted.lower().split()
        gt_words = ground_truth.lower().split()
        
        word_matches = 0
        for word in predicted_words:
            if word in gt_words:
                word_matches += 1
        
        word_accuracy = word_matches / len(gt_words) if len(gt_words) > 0 else 0.0
        
        return {
            'similarity_ratio': similarity_ratio,
            'normalized_edit_distance': normalized_edit_distance,
            'word_accuracy': word_accuracy,
            'character_accuracy': normalized_edit_distance,
            'edit_distance': edit_dist
        }
    
    def test_language_dataset(self, language: str) -> Dict[str, any]:
        """Test OCR on a specific language dataset"""
        print(f"\nTesting {language} dataset...")
        
        # Load ground truth
        gt_path = self.test_data_path / "ground_truth" / f"{language}_ground_truth.json"
        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        # Image directory
        img_dir = self.test_data_path / language
        
        results = {}
        total_metrics = {
            'similarity_ratio': 0,
            'normalized_edit_distance': 0,
            'word_accuracy': 0,
            'character_accuracy': 0,
            'edit_distance': 0
        }
        
        processed_count = 0
        
        for filename, gt_text in ground_truth.items():
            img_path = img_dir / filename
            
            if not img_path.exists():
                print(f"Warning: Image {filename} not found")
                continue
            
            print(f"Processing {filename}...")
            
            # Extract text using PaddleOCR
            start_time = time.time()
            predicted_text = self.extract_text_paddleocr(str(img_path), language)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self.calculate_similarity_metrics(predicted_text, gt_text)
            
            # Store results
            results[filename] = {
                'predicted': predicted_text,
                'ground_truth': gt_text,
                'metrics': metrics,
                'processing_time': processing_time
            }
            
            # Add to totals
            for metric in total_metrics:
                total_metrics[metric] += metrics[metric]
            
            processed_count += 1
        
        # Calculate averages
        avg_metrics = {k: v / processed_count for k, v in total_metrics.items()}
        
        return {
            'results': results,
            'average_metrics': avg_metrics,
            'processed_count': processed_count
        }
    
    def run_full_test(self) -> Dict[str, any]:
        """Run OCR test on all datasets"""
        print("Starting PaddleOCR Test")
        print("=" * 50)
        
        full_results = {}
        
        # Test both languages
        for language in ['english', 'french']:
            lang_results = self.test_language_dataset(language)
            full_results[language] = lang_results
            
            # Print summary for this language
            print(f"\n{language.upper()} RESULTS SUMMARY:")
            print("-" * 30)
            avg_metrics = lang_results['average_metrics']
            print(f"Files processed: {lang_results['processed_count']}")
            print(f"Average Similarity Ratio: {avg_metrics['similarity_ratio']:.4f}")
            print(f"Average Character Accuracy: {avg_metrics['character_accuracy']:.4f}")
            print(f"Average Word Accuracy: {avg_metrics['word_accuracy']:.4f}")
            print(f"Average Normalized Edit Distance: {avg_metrics['normalized_edit_distance']:.4f}")
        
        return full_results
    
    def save_detailed_results(self, results: Dict[str, any], output_file: str = "paddleocr_test_results.json"):
        """Save detailed results to JSON file"""
        output_path = Path(output_file)
        
        # Calculate overall statistics
        all_processing_times = []
        all_accuracies = []
        total_files = 0
        
        for lang_data in results.values():
            for result in lang_data['results'].values():
                all_processing_times.append(result['processing_time'])
                all_accuracies.append(result['metrics']['similarity_ratio'])
                total_files += 1
        
        avg_latency = sum(all_processing_times) / len(all_processing_times) if all_processing_times else 0
        avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
        
        # Get PaddleOCR version
        try:
            import paddleocr
            if hasattr(paddleocr, '__version__'):
                paddle_version = paddleocr.__version__
            else:
                paddle_version = "2.0+"  # Default recent version
        except:
            paddle_version = "unknown"
        
        # Trade-off information
        trade_off_log = {
            "model_info": {
                "name": "PaddleOCR",
                "version": paddle_version,
                "type": "Open-source",
                "deployment": "On-premise"
            },
            "accuracy_vs_latency": {
                "average_accuracy": avg_accuracy,
                "average_latency_seconds": avg_latency,
                "total_files_processed": total_files
            },
            "deployment_type": {
                "on_premise": True,
                "cloud": False,
                "hybrid": False
            },
            "licensing": {
                "open_source": True,
                "commercial": False,
                "license_type": "Apache 2.0"
            }
        }
        
        # Convert Path objects to strings for JSON serialization
        serializable_results = {
            "trade_off_log": trade_off_log,
            "detailed_results": {}
        }
        
        for lang, lang_data in results.items():
            serializable_results["detailed_results"][lang] = {
                'average_metrics': lang_data['average_metrics'],
                'processed_count': lang_data['processed_count'],
                'results': lang_data['results']
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_path}")
    
    def print_sample_comparisons(self, results: Dict[str, any], num_samples: int = 3):
        """Print sample comparisons for manual inspection"""
        print(f"\nSAMPLE COMPARISONS (first {num_samples} files):")
        print("=" * 80)
        
        for lang, lang_data in results.items():
            print(f"\n{lang.upper()} SAMPLES:")
            print("-" * 40)
            
            count = 0
            for filename, result in lang_data['results'].items():
                if count >= num_samples:
                    break
                
                print(f"\nFile: {filename}")
                print(f"Similarity: {result['metrics']['similarity_ratio']:.3f}")
                print(f"Ground Truth: {result['ground_truth'][:100]}...")
                print(f"Predicted:    {result['predicted'][:100]}...")
                print("-" * 40)
                
                count += 1

def main():
    print("PADDLEOCR TEST")
    print("=" * 80)
    
    # Check if PaddleOCR is installed
    try:
        import paddleocr
        if hasattr(paddleocr, '__version__'):
            version = paddleocr.__version__
        else:
            version = "2.0+"  # Default recent version
        print(f"PaddleOCR version: {version}")
    except ImportError:
        print("Error: PaddleOCR not found. Please install PaddleOCR.")
        print("Install with: pip install paddlepaddle paddleocr")
        return
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Initialize tester
    try:
        tester = PaddleOCRTester()
    except Exception as e:
        print(f"Error initializing PaddleOCR tester: {e}")
        return
    
    # Debug path information
    print(f"Script location: {Path(__file__).absolute()}")
    print(f"Looking for test data at: {tester.test_data_path.absolute()}")
    
    # Check if test data exists
    if not tester.test_data_path.exists():
        print(f"Error: Test data directory not found at {tester.test_data_path.absolute()}")
        print("Please verify the directory structure:")
        print("- prototype/")
        print("  - test_data/")
        print("    - english/")
        print("    - french/") 
        print("    - ground_truth/")
        print("  - test_ocr/")
        print("    - PaddleOCR.py")
        return
    
    print(f"Test data directory found successfully!")
    
    # Run tests
    try:
        results = tester.run_full_test()
        
        # Save results
        tester.save_detailed_results(results)
        
        # Print sample comparisons
        tester.print_sample_comparisons(results)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
