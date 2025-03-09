"""
REQUISITOS: 
pip install torch==2.3.0+cpu torchaudio==2.3.0+cpu demucs==4.0.1 tqdm pydub inquirer soundfile
"""

import os
import sys
import inquirer
import warnings
from pathlib import Path
from pydub import AudioSegment
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from tqdm import tqdm

# Configuración profesional
AUTO_SAMPLE_RATE = 44100
TORCH_THREADS = 1        

class SeparadorInteractivo:
    def __init__(self):
        # Configurar modelo para baja RAM
        self.modelo = get_model(name='htdemucs')
        self.modelo.to('cpu')
        self.modelo.segment = 10.0  
        self.shifts = 2  
        
        # Configurar
        torch.set_num_threads(TORCH_THREADS)
        os.environ["OMP_NUM_THREADS"] = str(TORCH_THREADS)
        
    def _convertir_audio(self, archivo):
        """Conversión inteligente a WAV con parámetros exactos"""
        try:
            temp_file = Path(archivo).with_suffix('.temp.wav')
            audio = AudioSegment.from_file(archivo)
            
            audio.export(
                temp_file,
                format="wav",
                codec="pcm_s16le",
                bitrate="256k",
                parameters=["-ac", "2", "-ar", str(AUTO_SAMPLE_RATE)]
            )
            
            # Cargar 
            waveform, sr = torchaudio.load(temp_file, backend="soundfile")
            waveform = waveform.unsqueeze(0)
            
            temp_file.unlink()
            return waveform, sr
        except Exception as e:
            print(f"🔥 Error en conversión: {str(e)}")
            raise
    
    def _procesar_archivo(self, archivo, instrumentos, modo_instrumental=False, vocal_volume=0.1):
        """Núcleo del procesamiento con IA"""
        try:
            # Convertir
            waveform, sr = self._convertir_audio(archivo)
            original = waveform.squeeze(0)  # Audio original
            
            # Procesar 
            with tqdm(total=100, desc="Procesando", ncols=80) as barra:
                def progress_callback(p):
                    barra.n = int(p * 100)
                    barra.refresh()
                
                pistas = apply_model(
                    self.modelo,
                    waveform,
                    progress=progress_callback,
                    shifts=self.shifts
                )
            
            # Guardar
            base_name = Path(archivo).stem
            if modo_instrumental:
                # Crear
                vocals = None
                for nombre, datos in zip(self.modelo.sources, pistas[0]):
                    if nombre == 'vocals':
                        vocals = datos.cpu()
                
                if vocals is not None:
                    min_length = min(original.shape[1], vocals.shape[1])
                    original = original[:, :min_length]
                    vocals = vocals[:, :min_length]
                    instrumental_puro = original - vocals  
                    vocales_reducidas = vocals * vocal_volume  
                    mezcla = instrumental_puro + vocales_reducidas  
                else:
                    mezcla = original  
                
                ruta = Path("salidas") / f"{base_name}_instrumental_con_fondo.wav"
                torchaudio.save(
                    str(ruta),
                    mezcla,
                    sr,
                    encoding="PCM_S",
                    bits_per_sample=16
                )
            else:
                # Modo normal
                for nombre, datos in zip(self.modelo.sources, pistas[0]):
                    if nombre in instrumentos:
                        ruta = Path("salidas") / f"{base_name}_{nombre}.wav"
                        torchaudio.save(
                            str(ruta),
                            datos.cpu(),
                            sr,
                            encoding="PCM_S",
                            bits_per_sample=16
                        )
            return True
        except Exception as e:
            print(f"\n❌ Error crítico en {Path(archivo).name}: {str(e)}")
            return False

    def interfaz_usuario(self):
        """Interfaz interactiva estilo Hollywood"""
        print("\n" + "="*50)
        print("🎧 MANZANO ES GAY SDP V-1.3")
        print("="*50)
        
        # Selección
        archivos = [f for f in Path("canciones").glob("*") if f.suffix.lower() in ('.mp3','.wav','.flac','.ogg')]
        if not archivos:
            input("\n🚨 Coloca archivos en la carpeta 'canciones' y presiona Enter...")
            return
        
        # Primera
        preguntas_iniciales = [
            inquirer.Checkbox('archivos',
                message="📁 Selecciona archivos a procesar (Espacio para marcar)",
                choices=[(f.name, str(f)) for f in archivos],
                carousel=True
            ),
            inquirer.List('modo',
                message="🎨 Selecciona el modo de salida",
                choices=['Pistas individuales', 'Instrumental con voces de fondo'],
                default='Pistas individuales'
            )
        ]
        respuestas_iniciales = inquirer.prompt(preguntas_iniciales)
        
        # Segunda
        if respuestas_iniciales['modo'] == 'Pistas individuales':
            preguntas_instrumentos = [
                inquirer.Checkbox('instrumentos',
                    message="🎛️ Instrumentos a extraer",
                    choices=['vocals', 'drums', 'bass', 'other'],
                    default=['vocals']
                )
            ]
            respuestas_instrumentos = inquirer.prompt(preguntas_instrumentos)
            instrumentos = respuestas_instrumentos['instrumentos']
        else:
            instrumentos = []  
        
        # Ajuste
        if respuestas_iniciales['modo'] == 'Instrumental con voces de fondo':
            vocal_volume = float(input("Ingresa el volumen de las voces (0.0 para excluir, 0.2 para incluir reducido): "))
        else:
            vocal_volume = 0.0
        
        # Procesamiento
        exito_total = 0
        modo_instrumental = respuestas_iniciales['modo'] == 'Instrumental con voces de fondo'
        for archivo in respuestas_iniciales['archivos']:
            print(f"\n⚡ Procesando: {Path(archivo).name}")
            if self._procesar_archivo(archivo, instrumentos, modo_instrumental, vocal_volume):
                exito_total += 1
                print(f"✅ {Path(archivo).name} completado!")
            else:
                print(f"🔥 {Path(archivo).name} falló")
        
        # Resultado 
        print(f"\n✨ ¡Proceso terminado! {exito_total}/{len(respuestas_iniciales['archivos'])} archivos exitosos")
        if os.name == 'nt':
            os.system("pause")

if __name__ == "__main__":
    # Verificar
    try:
        import soundfile
    except ImportError:
        print("🚨 Ejecuta esto primero:")
        print("pip install soundfile")
        sys.exit(1)
    
    # Crear
    Path("canciones").mkdir(exist_ok=True)
    Path("salidas").mkdir(exist_ok=True)
    
    # Iniciar
    separador = SeparadorInteractivo()
    separador.interfaz_usuario()
