# 🎶 Audio Super-Resolution with GANs  

**Audio Super-Resolution with GANs** enhances compressed MP3 audio (128, 256, 320 kbps) into high-quality FLAC files (~800+ kbps).  
Using adversarial learning, it restores lost high-frequency details and natural timbre, producing near-lossless results.  

---

## ✨ Features  
- Input: **MP3** (128/256/320 kbps)  
- Output: **FLAC** (~800+ kbps effective bitrate)  
- GAN-based model reconstructs missing harmonics & spectral detail  
- Applications: music remastering, streaming enhancement, archival recovery  

---

## ⚙️ How It Works  
1. **Generator** predicts high-resolution spectra from compressed input  
2. **Discriminator** enforces perceptual realism  
3. Training uses **adversarial + spectral + perceptual losses**  

---

## 🚀 Usage  

```bash
# Clone the repo
git clone https://github.com/rohan-prasen/Audio_Super-Res-Net.git
cd Audio_Super-Res-Net

# Install dependencies
pip install -r requirements.txt

# Run super-resolution
python test.py
