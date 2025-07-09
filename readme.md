OCR compare

To create data, run prototype\src\create_sample_data.py

Install package: `pip install -r requirements.txt`

## Tesseract OCR

Install Tesseract OCR [here](https://www.youtube.com/watch?v=2kWvk4C1pMo)

For French, install pretrained model from [here](https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata) then put it manual into `C:\Program Files\Tesseract-OCR\tessdata\`

run `python prototype\test_ocr\tesseract.py` to test

## Cloud Vision API

Enter [Google Cloud Console](https://console.cloud.google.com/)

Create new project

Turn API on: Cloud Vision API

Get API key and put in prototypes/.env

Cloud_vision_key = 'key_here'

## EasyOCR

run `prototype\test_ocr\easyOCR.py`

## Kraken

Download pretrained model from [here](https://zenodo.org/records/2577813/files/en_best.mlmodel?download=1) and put into prototypes/models/

run `prototype\test_ocr\kraken_ocr.py`
