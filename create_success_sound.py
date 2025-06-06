import numpy as np
from scipy.io import wavfile

# Configuración del sonido
sample_rate = 44100  # Hz
duration = 0.5  # segundos

# Crear un sonido de éxito (ding)
t = np.linspace(0, duration, int(sample_rate * duration))
frequency1 = 880  # Hz (A5)
frequency2 = 1320  # Hz (E6)

# Crear las ondas
wave1 = np.sin(2 * np.pi * frequency1 * t) * 0.5
wave2 = np.sin(2 * np.pi * frequency2 * t) * 0.3

# Combinar las ondas
combined_wave = wave1 + wave2

# Aplicar envolvente ADSR simple
attack = int(0.05 * sample_rate)
decay = int(0.1 * sample_rate)
sustain_level = 0.7
release = int(0.35 * sample_rate)

envelope = np.ones(len(t))
# Attack
envelope[:attack] = np.linspace(0, 1, attack)
# Decay
envelope[attack:attack+decay] = np.linspace(1, sustain_level, decay)
# Release
release_start = len(t) - release
envelope[release_start:] = np.linspace(sustain_level, 0, release)

# Aplicar envolvente
final_wave = (combined_wave * envelope).astype(np.float32)

# Normalizar
final_wave = final_wave / np.max(np.abs(final_wave))

# Guardar como archivo WAV
wavfile.write('assets/sounds/success.wav', sample_rate, final_wave) 