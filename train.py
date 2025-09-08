import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1, output_padding=(1, 1)),
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
            nn.Sigmoid()  # PatchGAN style – outputs patch-wise probabilities
        )
    
    def forward(self, x):
        return self.model(x)

# -----------------------------------
# Dataset: Load lossless (.flac) audio
# -----------------------------------
class LosslessAudioDataset(Dataset):
    def __init__(self, directory, segment_length=44100, hop_length=22050, sample_rate=44100):
        super().__init__()
        self.directory = directory
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.segments = []  # List of (file_path, start_sample)
        
        # Recursively scan for .flac files
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.flac'):
                    file_path = os.path.join(root, file)
                    info = torchaudio.info(file_path)
                    if info.sample_rate != self.sample_rate:
                        print(f"Skipping {file_path}: sample rate mismatch.")
                        continue
                    num_samples = info.num_frames
                    print(f"Found {file_path} with {num_samples} samples.")
                    # Create segments from the file
                    for start in range(0, num_samples - segment_length + 1, hop_length):
                        self.segments.append((file_path, start))
                    # If the file length is not exactly divisible, add the last segment
                    if num_samples > segment_length and (num_samples - segment_length) % hop_length != 0:
                        self.segments.append((file_path, num_samples - segment_length))
        print(f"Total segments found: {len(self.segments)}")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        file_path, start = self.segments[idx]
        waveform, sr = torchaudio.load(file_path)
        assert sr == self.sample_rate, f"Expected {self.sample_rate} Hz, got {sr} Hz"
        # Ensure stereo – if mono, expand to 2 channels
        if waveform.shape[0] != 2:
            waveform = waveform.expand(2, -1)
        end = start + self.segment_length
        if end > waveform.shape[1]:
            pad = end - waveform.shape[1]
            segment = torch.cat([waveform[:, start:], torch.zeros(2, pad)], dim=1)
        else:
            segment = waveform[:, start:end]
        return segment

# ------------------------------------------
# Updated function: Compute STFT magnitude for batch waveform of shape [B, 2, L]
def compute_stft_magnitude(waveform, n_fft=1024, hop_length=256):
    B, C, L = waveform.shape
    window = torch.hann_window(n_fft).to(waveform.device)
    # Reshape to (B * C, L) so that each channel is processed individually.
    waveform_reshaped = waveform.reshape(B * C, L)
    stft = torch.stft(waveform_reshaped, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    magnitude = stft.abs()  # Shape: (B * C, freq_bins, time_frames)
    # Reshape back to (B, C, freq_bins, time_frames)
    magnitude = magnitude.reshape(B, C, magnitude.size(1), magnitude.size(2))
    return magnitude

# ------------------------------
# Main Training Loop with Early Stopping
# ------------------------------
if __name__ == "__main__":
    lossless_dir = r'../src/data/train/hq'
    num_epochs = 100  
    batch_size = 4
    learning_rate = 2e-4
    lambda_l1 = 100 
    patience = 11 

    # Create dataset and dataloader
    dataset = LosslessAudioDataset(lossless_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize models
    generator = SEANetGenerator().to(device)
    discriminator = SEANetDiscriminator().to(device)
    
    # Loss functions and optimizers
    criterion_BCE = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    best_loss = float('inf')
    patience_counter = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, segment in enumerate(dataloader):
            segment = segment.to(device)  # (B, 2, segment_length)
            
            # Compute target (lossless) STFT magnitude
            target_mag = compute_stft_magnitude(segment)  # (B, 2, freq_bins, time_frames)
            
            # Simulate degradation by quantizing to 8-bit and dequantizing
            degraded_mag = torch.round(target_mag * 255) / 255
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            # Real examples
            real_preds = discriminator(target_mag)
            real_labels = torch.ones_like(real_preds)
            loss_D_real = criterion_BCE(real_preds, real_labels)
            
            # Fake examples (detach to avoid generator gradients)
            fake_mag = generator(degraded_mag)
            fake_preds = discriminator(fake_mag.detach())
            fake_labels = torch.zeros_like(fake_preds)
            loss_D_fake = criterion_BCE(fake_preds, fake_labels)
            
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()
            
            # --- Train Generator ---
            optimizer_G.zero_grad()
            # Adversarial loss (want discriminator to think fake is real)
            fake_preds_for_G = discriminator(fake_mag)
            loss_G_adv = criterion_BCE(fake_preds_for_G, real_labels)
            # Reconstruction (L1) loss
            loss_G_L1 = criterion_L1(fake_mag, target_mag)
            loss_G = loss_G_adv + lambda_l1 * loss_G_L1
            loss_G.backward()
            optimizer_G.step()
            
            epoch_loss += loss_G.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Generator Loss: {avg_epoch_loss:.4f}")
        
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            # Save best models
            os.makedirs("./checkpoints", exist_ok=True)
            torch.save(generator.state_dict(), "./checkpoints/best_generator.pth")
            torch.save(discriminator.state_dict(), "./checkpoints/best_discriminator.pth")
            print("Improved loss, saving model checkpoints.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break

    print("Training complete.")
