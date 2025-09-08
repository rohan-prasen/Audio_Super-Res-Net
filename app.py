import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import tempfile
import warnings

# ----------------------------------------------------------
# Suppress torchaudio backend warnings
# ----------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=r".*Torchaudio.*backend.*",
    category=UserWarning,
)

# ----------------------------
# Settings
# ----------------------------
SAMPLE_RATE = 44100
SEGMENT_LENGTH = 44100   # 1 second
HOP_LENGTH = 22050       # half overlap
MAX_DURATION = 150  # seconds to keep from uploaded file

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Define the GAN components
# ---------------------------
class SEANetGenerator(nn.Module):
    def __init__(self):
        super(SEANetGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out

class SEANetDiscriminator(nn.Module):
    def __init__(self):
        super(SEANetDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# ----------------------------------------------------------
# Helper function: Compute overlap-add normalization count
# ----------------------------------------------------------
def windowed_count(segment_length, hop_length_stft, device):
    n_fft = 1024
    window = torch.hann_window(n_fft).to(device)
    ones = torch.ones(2, segment_length, device=device)
    stft = torch.stft(ones, n_fft=n_fft, hop_length=hop_length_stft,
                      window=window, return_complex=True)
    return torch.istft(torch.ones_like(stft), n_fft=n_fft, hop_length=hop_length_stft,
                       window=window, length=segment_length)

# ----------------------------------------------------------
# Reconstruct audio
# ----------------------------------------------------------
def reconstruct_audio(waveform_mp3, generator):
    if waveform_mp3.shape[0] != 2:
        raise RuntimeError(f"Expected stereo (2 channels), got {waveform_mp3.shape[0]}")
    waveform_mp3 = waveform_mp3.to(device)

    num_samples = waveform_mp3.shape[1]
    n_fft = 1024
    hop_length_stft = 256
    window = torch.hann_window(n_fft).to(device)

    num_segments = max(1, (num_samples - SEGMENT_LENGTH) // HOP_LENGTH + 1)
    if (num_samples - SEGMENT_LENGTH) % HOP_LENGTH != 0:
        num_segments += 1

    reconstructed_waveform = torch.zeros_like(waveform_mp3).to(device)
    count = torch.zeros_like(waveform_mp3).to(device)

    generator.eval()
    with torch.no_grad():
        for i in range(num_segments):
            start = i * HOP_LENGTH
            end = start + SEGMENT_LENGTH
            if end > num_samples:
                pad = end - num_samples
                segment = torch.cat([waveform_mp3[:, start:], torch.zeros(2, pad, device=device)], dim=1)
            else:
                segment = waveform_mp3[:, start:end]

            stft = torch.stft(segment, n_fft=n_fft, hop_length=hop_length_stft,
                              window=window, return_complex=True)
            magnitude = stft.abs()
            phase = stft.angle()

            mag_in = magnitude.unsqueeze(0)
            gen_out = generator(mag_in).squeeze(0)

            # Resize if mismatch
            if gen_out.shape != magnitude.shape:
                gen_out = F.interpolate(gen_out.unsqueeze(0),
                                        size=(magnitude.shape[1], magnitude.shape[2]),
                                        mode='bilinear',
                                        align_corners=False).squeeze(0)

            generated_stft = gen_out * torch.exp(1j * phase)
            recon_segment = torch.istft(generated_stft, n_fft=n_fft, hop_length=hop_length_stft,
                                        window=window, length=SEGMENT_LENGTH)

            if end > num_samples:
                valid_len = num_samples - start
                reconstructed_waveform[:, start:] += recon_segment[:, :valid_len]
                count[:, start:] += windowed_count(valid_len, hop_length_stft, device)
            else:
                reconstructed_waveform[:, start:end] += recon_segment
                count[:, start:end] += windowed_count(SEGMENT_LENGTH, hop_length_stft, device)

    reconstructed_waveform = reconstructed_waveform / count.clamp(min=1)
    return reconstructed_waveform.cpu()

# ----------------------------------------------------------
# Evaluate with discriminator
# ----------------------------------------------------------
def evaluate_audio(reconstructed_waveform, discriminator):
    discriminator.eval()
    with torch.no_grad():
        stft_recon = torch.stft(reconstructed_waveform.to(device), n_fft=1024, hop_length=256,
                                window=torch.hann_window(1024).to(device), return_complex=True)
        magnitude_recon = stft_recon.abs()
        disc_score = discriminator(magnitude_recon.unsqueeze(0)).mean().item()
    return disc_score

# ----------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------
def main():

    st.set_page_config(page_title="Lossless Music Convertor", page_icon='music.png', layout='centered')

    st.title("🎵 SEANet Audio Reconstruction Player (Demo Version)")
    st.caption("Processing limited to first 150 seconds (stereo).")

    st.divider()

    uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])
    if uploaded_file is None:
        st.info("Please upload an MP3 file to begin.")
        return

    # Load models once
    @st.cache_resource
    def load_models():
        generator = SEANetGenerator().to(device)
        discriminator = SEANetDiscriminator().to(device)
        gen_ckpt = "best_generator.pth"
        disc_ckpt = "best_discriminator.pth"
        generator.load_state_dict(torch.load(gen_ckpt, map_location=device, weights_only=True))
        discriminator.load_state_dict(torch.load(disc_ckpt, map_location=device, weights_only=True))
        return generator, discriminator

    generator, discriminator = load_models()

    # Save uploaded file to temp dir
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
        tmp_mp3.write(uploaded_file.read())
        tmp_mp3_path = tmp_mp3.name

    # Load mp3 (soundfile backend)
    waveform, sr = torchaudio.load(tmp_mp3_path, backend="soundfile")
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Trim to MAX_DURATION seconds
    max_len = SAMPLE_RATE * MAX_DURATION
    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]

    st.subheader("▶ Original MP3")
    st.audio(tmp_mp3_path, format="audio/mp3")

    # Reconstruct
    with st.spinner("Reconstructing audio..."):
        reconstructed = reconstruct_audio(waveform, generator)

    # Save reconstructed flac (soundfile backend)
    tmp_flac = tempfile.NamedTemporaryFile(delete=False, suffix=".flac")
    torchaudio.save(tmp_flac.name, reconstructed, sample_rate=SAMPLE_RATE, format="flac", backend="soundfile")

    st.subheader("🎧 Reconstructed FLAC (first 150s)")
    st.audio(tmp_flac.name, format="audio/flac")

    # # Add download button
    # with open(tmp_flac.name, "rb") as f:
    #     st.download_button(
    #         label="⬇️ Download Reconstructed Audio (FLAC)",
    #         data=f,
    #         file_name=f"reconstructed{tmp_flac.name}.flac",
    #         mime="audio/flac"
    #     )

    # Get discriminator score
    disc_score = evaluate_audio(reconstructed, discriminator)
    st.metric("Discriminator Score", f"{disc_score*100:.2f}")
    st.markdown("lower the **better**")

if __name__ == "__main__":
    main()
