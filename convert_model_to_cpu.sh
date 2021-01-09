set -ex

wget -nc https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

rm -rf ViT-B-32

unzip ViT-B-32.pt

rsync -Pa ./clip/ViT-B-32-cpu/code/ ViT-B-32/code/

rm -f ViT-B-32-cpu.pt

zip -r ViT-B-32-cpu.pt ViT-B-32
