#!/bin/bash

echo "1. DANG TAI OLLAMA (2GB)..."
ollama pull llama3.2:3b

echo "2. DANG TAI CLIP (10GB)..."
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')"

echo "HOAN TAT!"
