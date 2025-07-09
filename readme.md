# OCR Compare

To create data, run:

```
prototype\src\create_sample_data.py
```

Install required packages:

```
pip install -r requirements.txt
```

## Tesseract OCR

Install Tesseract OCR:
[https://www.youtube.com/watch?v=2kWvk4C1pMo](https://www.youtube.com/watch?v=2kWvk4C1pMo)

For French, download the pretrained model:
[https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata](https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata)

Place the file manually into:

```
C:\Program Files\Tesseract-OCR\tessdata\
```

To test:

```
python prototype\test_ocr\tesseract.py
```

## Cloud Vision API

1. Go to:
   [https://console.cloud.google.com/](https://console.cloud.google.com/)

2. Create a new project

3. Enable the API: **Cloud Vision API**

4. Get your API key and add it to the `.env` file in the `prototypes/` folder:

```
Cloud_vision_key = 'key_here'
```

## EasyOCR

To test:

```
python prototype\test_ocr\easyOCR.py
```

## Kraken

Download the pretrained model:
[https://zenodo.org/records/2577813/files/en_best.mlmodel?download=1](https://zenodo.org/records/2577813/files/en_best.mlmodel?download=1)

Place it into:

```
prototypes/models/
```

To test:

```
python prototype\test_ocr\kraken_ocr.py
```

## PaddleOCR

To test:

```
python prototype\test_ocr\PaddleOCR.py
```
