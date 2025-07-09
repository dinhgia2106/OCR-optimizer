#!/usr/bin/env python3
"""
OCR Optimizer Prototype - Sample Data Creator
Creates synthetic test images with known text for evaluation
"""

import os
import sys
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# --- DATA FOR SAMPLE GENERATION ---

TEXT_FRAGMENTS = {
    'fr': {
        'templates': ["invoice", "report", "letter"],
        'greetings': ["Bonjour,", "Cher client,", "Madame, Monsieur,"],
        'companies': ["Innovatech SARL", "Solutions Globales SA", "Le Progrès Numérique", "Alpha Omega Inc."],
        'addresses': ["123 Rue de l'Avenir, 75001 Paris", "45 Avenue de la République, 69002 Lyon", "78 Boulevard de la Liberté, 13001 Marseille"],
        'invoice_items': [
            ("Consultation Stratégique", 1500), ("Développement Logiciel", 3200),
            ("Formation Utilisateur", 800), ("Support Technique Mensuel", 250),
            ("Licence Logiciel Annuelle", 1200)
        ],
        'report_titles': ["Rapport d'Activité Mensuel", "Analyse de Performance Trimestrielle", "Bilan Annuel 2025"],
        'report_body': [
            "Le chiffre d'affaires a augmenté de 12% ce trimestre.",
            "Nos initiatives marketing ont atteint une nouvelle audience cible.",
            "La satisfaction client reste notre priorité absolue, avec un score de 9.2/10.",
            "Nous prévoyons de lancer un nouveau produit au prochain semestre."
        ],
        'letter_body': [
            "Suite à notre conversation téléphonique, nous vous confirmons notre accord.",
            "Veuillez trouver ci-joint les documents demandés pour votre dossier.",
            "Nous restons à votre disposition pour toute information complémentaire."
        ],
        'closings': ["Cordialement,", "Sincères salutations,", "Bien à vous,"]
    },
    'en': {
        'templates': ["invoice", "report", "letter"],
        'greetings': ["Hello,", "Dear Customer,", "To Whom It May Concern,"],
        'companies': ["Innovatech LLC", "Global Solutions Corp.", "Digital Progress Co.", "Alpha Omega Inc."],
        'addresses': ["123 Future Street, New York, NY 10001", "45 Republic Avenue, Chicago, IL 60602", "78 Liberty Boulevard, Los Angeles, CA 90013"],
        'invoice_items': [
            ("Strategic Consultation", 1500), ("Software Development", 3200),
            ("User Training Session", 800), ("Monthly Technical Support", 250),
            ("Annual Software License", 1200)
        ],
        'report_titles': ["Monthly Activity Report", "Quarterly Performance Analysis", "Annual Review 2025"],
        'report_body': [
            "Revenue has increased by 12% this quarter.",
            "Our marketing initiatives have reached a new target audience.",
            "Customer satisfaction remains our top priority, with a score of 9.2/10.",
            "We plan to launch a new product in the next semester."
        ],
        'letter_body': [
            "Following our telephone conversation, we hereby confirm our agreement.",
            "Please find attached the requested documents for your file.",
            "We remain at your disposal for any further information."
        ],
        'closings': ["Sincerely,", "Best regards,", "Yours faithfully,"]
    }
}

def generate_text_samples(language: str, count: int = 20) -> dict:
    """Generates a dictionary of varied text samples."""
    samples = {}
    fragments = TEXT_FRAGMENTS[language]
    currency = "€" if language == 'fr' else "$"

    for i in range(count):
        doc_type = random.choice(fragments['templates'])
        text = ""

        if doc_type == "invoice":
            client = random.choice(fragments['companies'])
            address = random.choice(fragments['addresses'])
            num_items = random.randint(2, 4)
            items = random.sample(fragments['invoice_items'], num_items)
            total = sum(item[1] for item in items)
            
            text += f"Facture N° {random.randint(1000, 9999)}\n" if language == 'fr' else f"Invoice No. {random.randint(1000, 9999)}\n"
            text += f"Client: {client}\nAdresse: {address}\n\n" if language == 'fr' else f"Client: {client}\nAddress: {address}\n\n"
            text += "Description\t\tPrix\n" if language == 'fr' else "Description\t\tPrice\n"
            text += "--------------------------------\n"
            for desc, price in items:
                text += f"- {desc}: {price}{currency}\n"
            text += "--------------------------------\n"
            text += f"Total: {total}{currency}"

        elif doc_type == "report":
            title = random.choice(fragments['report_titles'])
            body_parts = random.sample(fragments['report_body'], random.randint(2, 3))
            
            text += f"{title}\n"
            text += f"Date: {random.randint(1,28)}/{random.randint(1,12)}/2025\n\n"
            text += "Points clés:\n" if language == 'fr' else "Key Points:\n"
            for part in body_parts:
                text += f"• {part}\n"

        elif doc_type == "letter":
            greeting = random.choice(fragments['greetings'])
            body = random.choice(fragments['letter_body'])
            closing = random.choice(fragments['closings'])
            sender = random.choice(fragments['companies'])

            text += f"{greeting}\n\n"
            text += f"{body}\n\n"
            text += f"{closing}\n\n{sender}"
        
        filename = f"sample_{i+1:02d}.jpg"
        samples[filename] = text

    return samples


def apply_noise_effects(img_array: np.ndarray) -> np.ndarray:
    """Applies a random combination of noise effects to an image array."""
    
    # 1. Gaussian Noise
    if random.random() < 0.7:
        mean = 0
        sigma = random.uniform(5, 15)
        noise = np.random.normal(mean, sigma, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Salt-and-Pepper Noise
    if random.random() < 0.4:
        s_vs_p = 0.5
        amount = random.uniform(0.001, 0.005)
        # Salt mode
        num_salt = np.ceil(amount * img_array.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
        img_array[coords[0], coords[1], :] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * img_array.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
        img_array[coords[0], coords[1], :] = 0

    # 3. Blur
    if random.random() < 0.8:
        kernel_size = random.choice([(3, 3), (5, 5)])
        img_array = cv2.GaussianBlur(img_array, kernel_size, 0)
        
    # 4. Brightness/Contrast
    if random.random() < 0.6:
        alpha = random.uniform(0.8, 1.2)  # Contrast control
        beta = random.uniform(-20, 20)     # Brightness control
        img_array = cv2.convertScaleAbs(img_array, alpha=alpha, beta=beta)

    # 5. Rotation
    if random.random() < 0.5:
        h, w = img_array.shape[:2]
        center = (w // 2, h // 2)
        angle = random.uniform(-2.5, 2.5)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_array = cv2.warpAffine(img_array, M, (w, h), borderValue=(255, 255, 255), flags=cv2.INTER_CUBIC)

    # 6. Perspective Warp
    if random.random() < 0.5:
        h, w = img_array.shape[:2]
        margin = int(min(h, w) * 0.05)
        
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # Perturb corners
        dx1 = random.uniform(-margin, margin)
        dy1 = random.uniform(-margin, margin)
        dx2 = random.uniform(-margin, margin)
        dy2 = random.uniform(-margin, margin)
        
        pts2 = np.float32([[dx1, dy1], [w - dx2, dy1], 
                           [dx1, h - dy2], [w - dx2, h - dy2]])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_array = cv2.warpPerspective(img_array, M, (w, h), borderValue=(255, 255, 255))
        
    return img_array


def create_text_image(text: str,
                     output_path: Path,
                     width: int = 800,
                     height: int = 1100, # Increased height for longer docs
                     font_size: int = 22,
                     background_color: tuple = (255, 255, 255),
                     text_color: tuple = (0, 0, 0),
                     add_noise: bool = True) -> bool:
    """
    Create a synthetic image with text and realistic noise.
    
    Args:
        text: Text to render
        output_path: Path to save the image
        width: Image width
        height: Image height
        font_size: Font size
        background_color: Background RGB color
        text_color: Text RGB color
        add_noise: Whether to add a combination of noise effects
        
    Returns:
        True if successful
    """
    try:
        image = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(image)
        
        # NOTE: Font path might need adjustment on different OS
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                # For macOS
                font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
            except IOError:
                try:
                    # For Linux (common path)
                    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
                except IOError:
                    # Fallback default font
                    font = ImageFont.load_default()
        
        margin = 50
        y_offset = margin
        line_height = font_size + 10

        # Split text into lines to handle newlines from generator
        text_lines = text.split('\n')
        
        for line in text_lines:
            if y_offset + line_height > height - margin:
                break
            draw.text((margin, y_offset), line, fill=text_color, font=font)
            y_offset += line_height
            
        img_array = np.array(image)
        
        if add_noise:
            img_array = apply_noise_effects(img_array)
        
        cv2.imwrite(str(output_path), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        return True
        
    except Exception as e:
        print(f"Error creating image {output_path}: {e}")
        return False

def create_sample_dataset():
    """Create sample dataset with French and English documents"""
    
    prototype_dir = Path(__file__).parent.parent
    test_data_dir = prototype_dir / "test_data"
    
    num_samples = 20
    
    # Generate sample texts
    french_texts = generate_text_samples('fr', num_samples)
    english_texts = generate_text_samples('en', num_samples)
    
    print("Creating sample test images...")
    
    # Create French images
    french_dir = test_data_dir / "french"
    french_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} French samples...")
    for filename, text in french_texts.items():
        output_path = french_dir / filename
        if create_text_image(text, output_path):
            print(f"  Created {filename}")
        else:
            print(f"  Failed to create {filename}")
    
    # Create English images
    english_dir = test_data_dir / "english"
    english_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} English samples...")
    for filename, text in english_texts.items():
        output_path = english_dir / filename
        if create_text_image(text, output_path):
            print(f"  Created {filename}")
        else:
            print(f"  Failed to create {filename}")
    
    # Update ground truth files
    ground_truth_dir = test_data_dir / "ground_truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    
    # Save French ground truth
    french_gt_file = ground_truth_dir / "french_ground_truth.json"
    cleaned_french = {k: ' '.join(v.split()) for k, v in french_texts.items()}
    
    with open(french_gt_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_french, f, ensure_ascii=False, indent=2)
    
    print(f"  Updated French ground truth: {french_gt_file}")
    
    # Save English ground truth
    english_gt_file = ground_truth_dir / "english_ground_truth.json"
    cleaned_english = {k: ' '.join(v.split()) for k, v in english_texts.items()}
    
    with open(english_gt_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_english, f, ensure_ascii=False, indent=2)
    
    print(f"  Updated English ground truth: {english_gt_file}")
    
    print("\nSample dataset creation completed!")
    print(f"French images: {len(french_texts)} files in {french_dir}")
    print(f"English images: {len(english_texts)} files in {english_dir}")
    print(f"Ground truth data saved in {ground_truth_dir}")
    
    return True

def main():
    """Main function"""
    print("OCR Sample Data Creator")
    print("=" * 50)
    
    try:
        create_sample_dataset()
    except Exception as e:
        print(f"Error creating sample data: {e}")
        return False
    
    print("\nSample data creation completed successfully!")
    print("\nNext steps:")
    print("1. Run: python src/demo.py --input test_data/french/sample_01.jpg")
    print("2. Run: python src/demo.py --batch test_data/french/ --output output/")
    print("3. Run: python src/evaluator.py --test_data test_data/french/ --ground_truth test_data/ground_truth/ --compare")
    
    return True

if __name__ == '__main__':
    main()