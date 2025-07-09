import cv2
import json
import os
import sys
from pathlib import Path
import difflib
from typing import Dict, List, Tuple
import re
import time
from dotenv import load_dotenv
import base64
import requests

# Add the parent directory to the path to access test_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CloudVisionTester:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get the correct path to test_data relative to this script
        current_dir = Path(__file__).parent
        self.test_data_path = current_dir.parent / "test_data"
        self.results = {}
        
        # Get API key from environment
        self.api_key = os.getenv('Cloud_vision_key')
        if not self.api_key:
            raise ValueError("Cloud_vision_key not found in .env file")
        
        # Cloud Vision API endpoint
        self.api_url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
        
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
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for Cloud Vision API"""
        # Use preprocessed image for better results
        processed_img = self.preprocess_image(image_path)
        
        # Encode processed image to base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
    
    def extract_text_cloud_vision(self, image_path: str, lang_hints: List[str] = None) -> str:
        """Extract text using Google Cloud Vision API"""
        try:
            # Encode image to base64
            image_base64 = self.encode_image_to_base64(image_path)
            
            # Prepare request payload
            request_payload = {
                "requests": [
                    {
                        "image": {
                            "content": image_base64
                        },
                        "features": [
                            {
                                "type": "DOCUMENT_TEXT_DETECTION",
                                "maxResults": 1
                            }
                        ],
                        "imageContext": {
                            "languageHints": lang_hints if lang_hints else ["en"]
                        }
                    }
                ]
            }
            
            # Make API request
            response = requests.post(
                self.api_url,
                json=request_payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                return ""
            
            result = response.json()
            
            # Extract text from response
            if 'responses' in result and result['responses']:
                response_data = result['responses'][0]
                if 'fullTextAnnotation' in response_data:
                    text = response_data['fullTextAnnotation']['text']
                    
                    # Clean extracted text
                    text = text.strip()
                    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with single space
                    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
                    
                    return text
                elif 'textAnnotations' in response_data and response_data['textAnnotations']:
                    # Fallback to textAnnotations if fullTextAnnotation is not available
                    text = response_data['textAnnotations'][0]['description']
                    
                    # Clean extracted text
                    text = text.strip()
                    text = re.sub(r'\n+', ' ', text)
                    text = re.sub(r'\s+', ' ', text)
                    
                    return text
            
            return ""
            
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
        
        # Set language hints for Cloud Vision
        lang_hints = ['en'] if language == 'english' else ['fr', 'en']
        
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
            
            # Extract text using Cloud Vision API
            start_time = time.time()
            predicted_text = self.extract_text_cloud_vision(str(img_path), lang_hints)
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
            
            # Add small delay to respect API rate limits
            time.sleep(0.1)
        
        # Calculate averages
        avg_metrics = {k: v / processed_count for k, v in total_metrics.items()}
        
        return {
            'results': results,
            'average_metrics': avg_metrics,
            'processed_count': processed_count
        }
    
    def run_full_test(self) -> Dict[str, any]:
        """Run OCR test on all datasets"""
        print("Starting Google Cloud Vision OCR Test")
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
    
    def save_detailed_results(self, results: Dict[str, any], output_file: str = "cloud_vision_test_results.json"):
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
        
        # Trade-off information
        trade_off_log = {
            "model_info": {
                "name": "Google Cloud Vision API",
                "version": "v1",
                "type": "Commercial",
                "deployment": "Cloud"
            },
            "accuracy_vs_latency": {
                "average_accuracy": avg_accuracy,
                "average_latency_seconds": avg_latency,
                "total_files_processed": total_files
            },
            "deployment_type": {
                "on_premise": False,
                "cloud": True,
                "hybrid": False
            },
            "licensing": {
                "open_source": False,
                "commercial": True,
                "license_type": "Proprietary"
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
    print("GOOGLE CLOUD VISION OCR TEST")
    print("=" * 80)
    
    # Check if .env file exists
    env_path = Path('.env')
    if not env_path.exists():
        print("Error: .env file not found. Please create .env file with Cloud_vision_key")
        return
    
    # Initialize tester
    try:
        tester = CloudVisionTester()
        print("Cloud Vision API key loaded successfully")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please add Cloud_vision_key to your .env file")
        return
    except Exception as e:
        print(f"Error initializing Cloud Vision tester: {e}")
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
        print("    - CloudVision.py")
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
