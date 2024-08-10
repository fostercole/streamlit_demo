import os
import zipfile
import pandas as pd
import torch
import warnings
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

def extract_zip(zip_path, extract_to):
    """Extracts a ZIP file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def compute_image_embeddings(image_dir, model, processor):
    """Converts images in a directory to embeddings using the CLIP model."""
    image_embeddings = []

    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path).convert('RGB')

            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                image_embeddings.append(outputs)

    if image_embeddings:
        return torch.cat(image_embeddings)
    else:
        return None

def compute_text_embeddings(csv_path, model, processor, text_column):
    """Converts text in a CSV file to embeddings using the CLIP model."""
    text_embeddings = []

    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        print(f"Column '{text_column}' not found in {csv_path}. Available columns: {df.columns}")
        return None

    for text in df[text_column]:
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            text_embeddings.append(outputs)

    if text_embeddings:
        return torch.cat(text_embeddings)
    else:
        return None

def save_tensor(tensor, file_path):
    """Saves a PyTorch tensor to a file."""
    if tensor is not None:
        torch.save(tensor, file_path)
        print(f'Saved tensor to {file_path}')
    else:
        print(f'No tensor to save for {file_path}')

def main():
    # Define file paths with correct names and spaces
    csv_files = {
        'all_kindle_review .csv': 'reviewText',  # Adjust column name if necessary
        'amazonFood.csv': 'Text',  # Updated to correct column name
        'spanish_hotel_reviews.csv': 'review',  # Adjust column name if necessary
    }

    zip_files = [
        'chest_xray_images.zip',
        'pokemon_images.zip',
        'stop_signs.zip',
    ]

    embeddings_dir = 'embeddings'
    os.makedirs(embeddings_dir, exist_ok=True)

    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Compute and save text embeddings for each CSV file
    for csv_file, text_column in csv_files.items():
        if os.path.exists(csv_file):
            print(f'Processing {csv_file}...')
            text_embeddings = compute_text_embeddings(csv_file, model, processor, text_column)
            csv_embeddings_path = os.path.join(embeddings_dir, f'{os.path.splitext(csv_file)[0]}_embeddings.pt')
            save_tensor(text_embeddings, csv_embeddings_path)
        else:
            print(f'{csv_file} does not exist.')

    # Compute and save image embeddings for each ZIP file
    for zip_file in zip_files:
        if os.path.exists(zip_file):
            print(f'Processing {zip_file}...')
            extract_to = os.path.join('extracted', os.path.splitext(zip_file)[0])
            os.makedirs(extract_to, exist_ok=True)
            extract_zip(zip_file, extract_to)

            image_embeddings = compute_image_embeddings(extract_to, model, processor)
            image_embeddings_path = os.path.join(embeddings_dir, f'{os.path.splitext(zip_file)[0]}_embeddings.pt')
            save_tensor(image_embeddings, image_embeddings_path)
        else:
            print(f'{zip_file} does not exist.')

if __name__ == '__main__':
    main()
