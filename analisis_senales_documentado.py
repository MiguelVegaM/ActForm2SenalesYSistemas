
# ===============================================
# ANÁLISIS DE SEÑALES EN DOMINIO DEL TIEMPO Y FRECUENCIA
# Autor: Miguel Vega
# Descripción: Este script genera señales básicas, calcula su Transformada de Fourier
# usando np.fft.fft() y verifica propiedades como linealidad, desplazamiento y escalamiento.
# ===============================================

import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1. DEFINICIÓN DE SEÑALES
# ===========================

# Creamos un vector de tiempo de -1 a 1 con 500 puntos
t = np.linspace(-1, 1, 500)

# Pulso rectangular centrado en t=0 con duración 0.4s
pulse = np.where((t >= -0.2) & (t <= 0.2), 1, 0)

# Función escalón unitario (Heaviside) en t=0
step = np.where(t >= 0, 1, 0)

# Señal senoidal de 5 Hz
f = 5  # frecuencia en Hz
sine = np.sin(2 * np.pi * f * t)

# ===========================
# 2. GRAFICAR SEÑALES EN TIEMPO
# ===========================

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t, pulse)
plt.title('Pulso Rectangular')
plt.ylabel('Amplitud')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, step)
plt.title('Función Escalón')
plt.ylabel('Amplitud')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, sine)
plt.title('Señal Senoidal (5 Hz)')
plt.ylabel('Amplitud')
plt.xlabel('Tiempo (s)')
plt.grid()

plt.tight_layout()
plt.show()

# ===========================
# 3. FUNCIÓN PARA GRAFICAR FFT
# ===========================

def plot_fft(signal, t, title=''):
    """
    Calcula y grafica la FFT de una señal dada.
    Muestra la magnitud y la fase del espectro de frecuencia.
    """
    N = len(signal)
    fft_vals = np.fft.fft(signal)              # Transformada de Fourier
    fft_freqs = np.fft.fftfreq(N, d=(t[1] - t[0]))  # Eje de frecuencias

    magnitude = np.abs(fft_vals)
    phase = np.angle(fft_vals)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.stem(fft_freqs, magnitude, use_line_collection=True)
    plt.title(f'Magnitud - {title}')
    plt.xlabel('Frecuencia (Hz)')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.stem(fft_freqs, phase, use_line_collection=True)
    plt.title(f'Fase - {title}')
    plt.xlabel('Frecuencia (Hz)')
    plt.grid()

    plt.tight_layout()
    plt.show()

# ===========================
# 4. GRAFICAR FFT DE SEÑALES BÁSICAS
# ===========================

plot_fft(pulse, t, title='Pulso Rectangular')
plot_fft(step, t, title='Función Escalón')
plot_fft(sine, t, title='Señal Senoidal (5 Hz)')

# ===========================
# 5. VERIFICACIÓN DE PROPIEDADES
# ===========================

# ---- Linealidad: suma de dos señales ----
linear_combo = pulse + sine
plot_fft(linear_combo, t, title='Linealidad (Pulso + Senoidal)')

# ---- Desplazamiento en el tiempo: corrimiento del pulso ----
pulse_shifted = np.roll(pulse, 50)  # Corrimiento de 50 muestras
plot_fft(pulse_shifted, t, title='Pulso Desplazado en el Tiempo')

# ---- Escalamiento en frecuencia: aumentar la frecuencia de la senoidal ----
sine_high_freq = np.sin(2 * np.pi * 10 * t)  # 10 Hz
plot_fft(sine_high_freq, t, title='Senoidal 10 Hz (Escalamiento en Frecuencia)')

# ===============================================
# Fin del script
# ===============================================
