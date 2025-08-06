<!-- install torch -->
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

<!-- install flash attn -->
pip install flash-attn

<!-- install transformers -->
pip install transformers

<!-- install qwen-vl-utils -->
pip install qwen-vl-utils[decord]

then install fastapi, rich, accelerate, bitsandbytes, python-multipart