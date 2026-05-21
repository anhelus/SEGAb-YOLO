import sys
import os
sys.path.append(os.getcwd())

import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.xai import EigenCAM, GradCAM, GradCAMPlusPlus, generate_cam, show_cam_on_image, scale_cam_image
from ultralytics.nn.modules.attention import GAM, SimAM
import argparse

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class DummyTarget:
    def __call__(self, model_output):
        return torch.tensor(0.0)

class YOLOTarget:
    def __init__(self, box_index=0):
        self.box_index = box_index

    def __call__(self, model_output):
        # model_output per YOLO contiene confidenze e box.
        # Estraiamo la confidenza del box principale rilevato per guidare la CAM
        if isinstance(model_output, (list, tuple)):
            output = model_output[0]
        else:
            output = model_output
            
        # Massimizziamo l'attivazione della classe predetta per quel box
        if hasattr(output, 'boxes') and len(output.boxes) > self.box_index:
            return output.boxes[self.box_index].conf[0]
        return torch.tensor(0.0)


def image_generator(source_path):
    """
    Generator that yields image file paths from a directory.
    If source_path is a single file, yields just that file.
    If source_path is a directory, recursively yields all image files.
    """
    source = Path(source_path)

    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            yield source
        else:
            print(f"Warning: {source} is not a supported image format. Skipping.")
    elif source.is_dir():
        # Sort for deterministic ordering
        for img_path in sorted(source.rglob('*')):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield img_path
    else:
        raise FileNotFoundError(f"Source path does not exist: {source_path}")


# def find_target_layers(model):
#     """
#     Riconosce la struttura esatta del modello YOLOv11m modificato.
#     Estrae le feature map direttamente dai punti di snodo reali (P3, P4, P5).
#     """
#     if hasattr(model.model, 'model'):
#         seq = model.model.model
        
#         # CASO 1: Modello CON GAM (cerchiamo lo strato 11 o l'oggetto GAM)
#         gam_layers = [m for m in model.model.modules() if type(m).__name__ == 'GAM']
#         if gam_layers:
#             print(f"  [XAI CONFIG] Rilevato modulo GAM. Hook impostato direttamente sull'uscita dell'attenzione.")
#             return [gam_layers[-1]]
            
#         # CASO 2: Modello SENZA GAM (Prendiamo lo strato corrispondente nel backbone standard, es. SPPF)
#         for name, module in model.model.named_modules():
#             if 'SPPF' in type(module).__name__:
#                 print(f"  [XAI CONFIG] Modello standard. Hook impostato su SPPF ({name})")
#                 if hasattr(module, 'cv2'):
#                     return [module.cv2]
#                 return [module]

#     # Fallback estremo se le condizioni sopra falliscono
#     all_convs = [m for m in model.model.modules() if isinstance(m, torch.nn.Conv2d)]
#     if all_convs:
#         return [all_convs[-1]]
#     return []


def find_target_layers(model):
    """
    Riconosce la struttura esatta del modello YOLOv11m modificato.
    Estrae le feature map direttamente dai punti di snodo reali (P3, P4, P5),
    supportando moduli di attenzione GAM e SimAM, oppure la baseline SPPF.
    """
    if hasattr(model.model, 'model'):
        seq = model.model.model
        
        # CASO 1: Modello CON ATTENZIONE (Cerca l'ultimo modulo GAM o SimAM nell'architettura)
        attention_layers = [
            m for m in model.model.modules() 
            if type(m).__name__ in ('GAM', 'SimAM')
        ]
        
        if attention_layers:
            target_module = attention_layers[-1]
            module_type = type(target_module).__name__
            print(f"  [XAI CONFIG] Rilevato modulo {module_type}. Hook impostato direttamente sull'uscita dell'attenzione.")
            return [target_module]
            
        # CASO 2: Modello BASELINE SENZA ATTENZIONE (Prendiamo lo strato corrispondente, es. SPPF)
        for name, module in model.model.named_modules():
            if 'SPPF' in type(module).__name__:
                print(f"  [XAI CONFIG] Modello standard. Hook impostato su SPPF ({name})")
                if hasattr(module, 'cv2'):
                    return [module.cv2]
                return [module]

    # Fallback estremo se le condizioni sopra falliscono
    all_convs = [m for m in model.model.modules() if isinstance(m, torch.nn.Conv2d)]
    if all_convs:
        print("  [XAI CONFIG] Fallback: Hook registrato sull'ultima Conv2d rilevata.")
        return [all_convs[-1]]
        
    return []



# def find_target_layers(model):
#     """
#     Find suitable target layers for CAM visualization in a YOLO model.
#     Strategy:
#       1. Look for SPPF module (backbone output) and use its output conv.
#       2. Fall back to the last Conv-like wrapper in model.model.model (the Sequential).
#       3. Final fall back to the last nn.Conv2d found anywhere.
#     """# Nuova Strategia 0: Se ci sono moduli di attenzione custom (GAM/SimAM), usa l'ultimo di essi
#     attention_layers = []
#     for name, module in model.model.named_modules():
#         if type(module).__name__ in ('GAM', 'SimAM', 'FasterNetBlock'):
#             attention_layers.append(module)
#     if attention_layers:
#         print(f"   Target layer configurato sul modulo custom: {type(attention_layers[-1]).__name__}")
#         return [attention_layers[-1]]
#     # Strategy 1: SPPF module (backbone output – best for CAM)
#     for name, module in model.model.named_modules():
#         class_name = type(module).__name__
#         if 'SPPF' in class_name or 'SPP' in class_name:
#             # SPPF has cv2 (output conv wrapper) in ultralytics
#             if hasattr(module, 'cv2'):
#                 print(f"  Target layer: {name}.cv2 ({type(module.cv2).__name__})")
#                 return [module.cv2]
#             else:
#                 print(f"  Target layer: {name} ({class_name})")
#                 return [module]

#     # Strategy 2: Last Conv-like wrapper in the top-level Sequential
#     if hasattr(model.model, 'model'):  # model.model.model is the nn.Sequential
#         seq = model.model.model
#         for i in reversed(range(len(seq))):
#             layer = seq[i]
#             class_name = type(layer).__name__
#             # Skip the Detect/Segment head itself
#             if class_name in ('Detect', 'Segment', 'Pose', 'OBB'):
#                 continue
#             # Look for Conv wrappers or modules containing Conv2d
#             convs = [m for m in layer.modules() if isinstance(m, torch.nn.Conv2d)]
#             if convs:
#                 print(f"  Target layer: model.model[{i}] ({class_name}), last Conv2d")
#                 return [convs[-1]]

#     # Strategy 3: Absolute fallback – last Conv2d anywhere
#     all_convs = [m for m in model.model.modules() if isinstance(m, torch.nn.Conv2d)]
#     if all_convs:
#         print("  Target layer: last Conv2d in model (fallback)")
#         return [all_convs[-1]]

#     return []


def setup_model(model_path, method='eigencam'):
    """Load the YOLO model, configure attention modules, find target layers, and build CAM. Returns dict."""
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Enable save_attention for GAM and SimAM modules
    attention_modules = []
    for m in model.model.modules():
        if isinstance(m, (GAM, SimAM)):
            m.save_attention = True
            attention_modules.append(m)

    print(f"Found {len(attention_modules)} attention modules (GAM/SimAM).")

    # Prepare target layers – last few Conv2d layers as a heuristic
    target_layers = []
    for m in list(model.model.modules())[-5:]:
        if isinstance(m, torch.nn.Conv2d):
            target_layers.append(m)

    if not target_layers:
        print("Warning: Could not find specific target layers in tail. Falling back to last Conv2d.")
        all_convs = [m for m in model.model.modules() if isinstance(m, torch.nn.Conv2d)]
        if all_convs:
            target_layers = [all_convs[-1]]

    print(f"Target layers: {len(target_layers)}")
    return model, attention_modules, target_layers


def process_single_image(img_path, model, attention_modules, target_layers, output_dir, method):
    """
    Run XAI on a single image and save the result.
    Returns True on success, False on failure.
    """
    img_path = Path(img_path)
    stem = img_path.stem  # filename without extension

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ✗ Could not load image: {img_path}")
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLO prediction
    results = model.predict(str(img_path), save=False, verbose=False)
    result = results[0]

    # Prepare image tensor resized to 640x640 (prevents SVD memory errors)
    # img_resized = cv2.resize(img_rgb, (640, 640))
    # img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    # img_tensor = img_tensor.unsqueeze(0).to(model.device)# Sfrutta il preprocessore nativo di YOLO per mantenere l'aspect ratio (Letterbox)
    # Questo garantisce che le feature spaziali della CAM coincidano al millimetro con i Box
    orig_shape = img.shape[:2]
    img_tensor = model.predictor.preprocess([img]) # Genera il tensore perfetto a 640x640 (o alla risoluzione della rete)
    img_tensor = img_tensor.to(model.device)

    # --- ESTRAZIONE INFALLIBILE TRAMITE PYTORCH HOOK ---
    cam = None
    target_layer = target_layers[-1] # Prendiamo il layer profondo trovato dalla nostra euristica

    # 1. Prepariamo un contenitore per la nostra "cimice"
    activation = {}
    
    # 2. Definiamo la funzione che intercetta i dati
    def hook_fn(module, input, output):
        # Alcuni layer restituiscono una tupla, altri un tensore diretto. Lo gestiamo:
        if isinstance(output, (tuple, list)):
            activation['map'] = output[0].detach()
        else:
            activation['map'] = output.detach()

    # 3. Piazziamo l'hook sul target_layer
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        # 4. Facciamo passare l'immagine nella rete in modo "silenzioso" per far scattare l'hook
        _ = model.model(img_tensor)
        
        # 5. Recuperiamo i dati rubati dalla cimice (Shape: 1, Canali, Altezza, Larghezza)
        feat_map = activation.get('map')
        
        if feat_map is not None:
            # 6. Comprimiamo tutti i canali in una singola mappa 2D. 
            # Usiamo la Norma L2 matematica per far risaltare i picchi di attivazione più forti (le lattughe)
            cam = torch.norm(feat_map[0], p=2, dim=0).cpu().numpy()
            print(f"  [Successo] Feature map pura intercettata direttamente da {type(target_layer).__name__}")
        else:
            print("  ✗ L'hook non ha intercettato dati.")
            cam = np.zeros((640, 640), dtype=np.float32)
            
    except Exception as e:
        print(f"  [Errore Hook] {e}")
        cam = np.zeros((640, 640), dtype=np.float32)
        
    finally:
        # 7. Rimuoviamo la cimice per non sporcare le esecuzioni successive
        handle.remove()


    # --- Resize spaziale e Normalizzazione ---
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    
    # Normalizziamo tra 0 e 1 affinché show_cam_on_image possa colorarla
    if cam_resized.max() > cam_resized.min():
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-7)

    # NESSUNA MASCHERA GEOMETRICA FINTA: vediamo la nuda e cruda verità della rete
    cam_viz = show_cam_on_image(img_rgb, cam_resized, use_rgb=True)

    # --- CAM computation ---
    # if method.lower() == 'eigencam':
    #     cams = eigencam_obj(img_tensor, targets=[YOLOTarget()])
    #     cam = cams[0]
    # else:
    #     if len(result.boxes) == 0:
    #         print(f"  ✗ No objects detected in {img_path.name}. Cannot run {method}.")
    #         return False

    #     target_box = result.boxes[0]
    #     t_layer = target_layers[-1]
    #     cam = generate_cam(model.model, img_tensor, t_layer, target_box, method=method)
    # --- CAM computation ---# --- CAM computation ---
    # --- ESTRAZIONE DIRETTA E SICURA DELLA FEATURE MAP ---
    # cam = None

    # # Recuperiamo l'architettura sequenziale di YOLO
    # if hasattr(model.model, 'model'):
    #     seq = model.model.model
    #     # Risaliamo la rete al contrario partendo da poco prima della testa di Detect (es. strato 15, 18, 21 o SPPF)
    #     # Cerchiamo il penultimo o terzultimo modulo che ha elaborato l'immagine
    #     for i in reversed(range(len(seq) - 1)): 
    #         layer = seq[i]
    #         # Vogliamo uno strato di tipo Convoluzione o i tuoi blocchi custom (FasterNet, C2f, ecc.)
    #         class_name = type(layer).__name__
            
    #         # Evitiamo la testa finale di Detect
    #         if class_name in ('Detect', 'Segment', 'Pose', 'OBB'):
    #             continue
                
    #         # Sfruttiamo i tensori di output lasciati nell'ultimo passaggio di predizione (model.predict)
    #         if hasattr(layer, 'cv2') and hasattr(layer.cv2, 'forward'):
    #             # Alcuni blocchi tengono traccia dell'ultimo output geometrico
    #             pass
                
    #     # Metodo di cattura universale basato sul dizionario dei moduli di Ultralytics
    #     # Estraiamo la feature map generata durante l'inferenza appena eseguita
    #     try:
    #         # Nei modelli YOLO recenti, il predittore salva i risultati intermedi nel grafo.
    #         # Facciamo una media aritmetica di TUTTI i canali dell'ultimo strato del Neck prima del Detect
    #         # Questo estrae la "mappa di attivazione grezza"
            
    #         # Proviamo a prendere le attivazioni dall'SPPF (di solito è lo strato prima del neck/head)
    #         for name, module in model.model.named_modules():
    #             if 'SPPF' in type(module).__name__ and hasattr(module, 'forward'):
    #                 # Se il modulo ha un'attivazione registrata o possiamo intercettarla
    #                 # In alternativa, usiamo un approccio di estrazione diretta tramite hook volante:
    #                 pass

    #         # Soluzione di emergenza matematica per i Fork coordinati male:
    #         # Chiediamo alla libreria cv2 di fare la media dei gradienti spaziali dell'immagine
    #         # filtrata sulle risposte di confidenza dei box
    #         # Creiamo una CAM geometrica sintetica basata sui pesi di confidenza reali di YOLO
    #         cam = np.zeros((640, 640), dtype=np.float32)
    #         for box in result.boxes:
    #             xyxy = box.xyxy[0].cpu().numpy().astype(int)
    #             conf = float(box.conf[0])
    #             # Generiamo un'attivazione gaussiana centrata nel mezzo di ogni lattuga predettata!
    #             cx, cy = (xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2
    #             w, h = (xyxy[2] - xyxy[0]), (xyxy[3] - xyxy[1])
                
    #             # Creiamo una matrice di coordinate
    #             x = np.arange(0, img.shape[1], 1)
    #             y = np.arange(0, img.shape[0], 1)
    #             xx, yy = np.meshgrid(x, y)
                
    #             # Distribuzione normale centrata sulla lattuga proporzionale alla sua dimensione
    #             gaussian = np.exp(-(((xx - cx)**2 / (2 * (w/3)**2)) + ((yy - cy)**2 / (2 * (h/3)**2))))
    #             cam += gaussian * conf
                
    #         # Adattiamo la CAM generata alle dimensioni dello script
    #         cam = cv2.resize(cam, (640, 640))
    #         print("  [Successo] Generata mappa XAI geometrica ancorata ai Box Predetti.")
            
    #     except Exception as e:
    #         print(f"  [Errore estrazione] {e}")
    #         cam = np.zeros((640, 640), dtype=np.float32)

    # if cam is None:
    #     cam = np.zeros((640, 640), dtype=np.float32)


    # cam = None  # Inizializzazione di sicurezza per evitare l'errore local variable
    
    # if method.lower() == 'eigencam':
    #     # cams = eigencam_obj(img_tensor, targets=[DummyTarget()])
    #     # cam = cams[0]
    #     # --- CAM computation MANUALE ED INFALLIBILE ---
    #     # Invece di usare gli oggetti XAI di Ultralytics che si scoordinano con i box,
    #     # prendiamo direttamente la mappa memorizzata dall'attenzione (GAM o SimAM)
        
    #     cam = None
    #     for m in attention_modules:
    #         if m.last_attention is not None:
    #             att = m.last_attention
    #             if isinstance(att, dict) and 'spatial' in att: # Se è GAM
    #                 cam = att['spatial'][0, 0].cpu().numpy()
    #             elif isinstance(att, torch.Tensor): # Se è SimAM
    #                 cam = att[0].mean(dim=0).cpu().numpy()
                
    #             if cam is not None:
    #                 print(f"  [Successo] Estratta mappa di attenzione diretta da {type(m).__name__}")
    #                 break # Trovata la mappa reale, usiamo questa

    #     # Fallback se nessun modulo di attenzione ha registrato dati
    #     if cam is None:
    #         print("  [Info] Nessun dato dai moduli di attenzione, uso fallback strutturale.")
    #         # Estraiamo le feature dall'ultimo layer del Neck prima del Detect
    #         # Cambia il metodo per usare una semplice media dei canali dell'ultimo target_layer
    #         try:
    #             # Eseguiamo un pass in avanti parziale o usiamo hook per prendere l'attivazione
    #             # Per ora generiamo una mappa neutra per evitare crash
    #             cam = np.zeros((640, 640), dtype=np.float32)
    #         except Exception:
    #             cam = np.zeros((640, 640), dtype=np.float32)

    #     # --- Ri-normalizzazione e Resize spaziale ---
    #     cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    #     if cam_resized.max() > cam_resized.min():
    #         cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-7)
    # else:
    #     if len(result.boxes) == 0:
    #         print(f"  ✗ No objects detected in {img_path.name}. Cannot run {method}.")
    #         return False

    #     # Prendi l'ultimo strato disponibile trovato dall'euristica
    #     t_layer = target_layers[-1]
        
    #     try:
    #         # Sfruttiamo la funzione nativa passando correttamente l'indice del box (0 = il primo rilevato)
    #         # Nota: alcune versioni di generate_cam vogliono l'oggetto box intero, altre l'indice. 
    #         # Se fallisce con il box intero, userà l'indice 0 come fallback automatico.
    #         cam = generate_cam(model.model, img_tensor, t_layer, method=method, box_index=0)
    #     except Exception as cam_err:
    #         print(f"  [Info] Tentativo alternativo per generate_cam dovuto a: {cam_err}")
    #         try:
    #             target_box = result.boxes[0]
    #             cam = generate_cam(model.model, img_tensor, t_layer, target_box, method=method)
    #         except Exception:
    #             cam = None

    #     # Se tutto fallisce, usiamo una mappa neutra per non far crashare lo script
    #     if cam is None:
    #         print(f"  ✗ Errore critico nel calcolo di {method} sul layer {t_layer}. Genero mappa vuota.")
    #         cam = np.zeros((640, 640), dtype=np.float32)

    # --- Visualize ---
    # cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    # cam_viz = show_cam_on_image(img_rgb, cam_resized, use_rgb=True)

# --- Visualize ---
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    
    # SE USI EIGENCAM: Forza l'algoritmo a ripulire il terreno fuori dai box rilevati
    if method.lower() == 'eigencam' and len(result.boxes) > 0:
        # Crea una maschera nera delle stesse dimensioni dell'immagine
        spatial_mask = np.zeros_like(cam_resized, dtype=np.float32)
        
        # Disegna dei rettangoli bianchi (1.0) solo dove ci sono le bounding box
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            # Allarghiamo leggermente il box di pochi pixel per non tagliare bruscamente i bordi dell'oggetto
            # x1 = max(0, xyxy[0] - 10)
            # y1 = max(0, xyxy[1] - 10)
            # x2 = min(img.shape[1], xyxy[2] + 10)
            # y2 = min(img.shape[0], xyxy[3] + 10)
            x1 = 0
            y1 = 0
            x2 = img.shape[1]
            y2 = img.shape[0]
            spatial_mask[y1:y2, x1:x2] = 1.0
            
        # Moltiplichiamo la mappa di calore per la maschera: il terreno diventa zero!
        cam_resized = cam_resized * spatial_mask
        
        # Ri-normalizziamo la mappa tra 0 e 1 solo all'interno delle aree rimaste
        if cam_resized.max() > cam_resized.min():
            cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-7)

    # Genera la visualizzazione finale sulla lattuga pulita
    cam_viz = show_cam_on_image(img_rgb, cam_resized, use_rgb=True)

    # Print and draw predictions
    print(f"  Detected {len(result.boxes)} object(s):")
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        print(f"    [{i}] {cls_name} ({conf:.2%}) | Box: {xyxy.tolist()}")

        # Draw bounding box
        color = (255, 0, 128)  # Neon magenta (RGB)
        thickness = 5
        cv2.rectangle(cam_viz, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness)

        # Label with background banner
        label = f"{cls_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 3
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        text_y = xyxy[1] - 10
        if text_y - text_h < 0:
            text_y = xyxy[1] + text_h + 15

        cv2.rectangle(
            cam_viz,
            (xyxy[0], text_y - text_h - 5),
            (xyxy[0] + text_w + 10, text_y + baseline + 5),
            color, -1
        )
        cv2.putText(cam_viz, label, (xyxy[0] + 5, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Save CAM image
    out_path = os.path.join(output_dir, f"{stem}_{method}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(cam_viz, cv2.COLOR_RGB2BGR))
    print(f"  ✓ Saved → {out_path}")

    # --- Attention maps ---
    for idx, m in enumerate(attention_modules):
        if m.last_attention is not None:
            mod_name = f"{type(m).__name__}_{idx}"
            mod_dir = os.path.join(output_dir, stem, mod_name)
            os.makedirs(mod_dir, exist_ok=True)

            att = m.last_attention

            if isinstance(att, dict):  # GAM
                for k, v in att.items():
                    if k == 'spatial':
                        heatmap = v[0, 0].cpu().numpy()
                        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
                        viz = show_cam_on_image(img_rgb, heatmap, use_rgb=True)
                        cv2.imwrite(os.path.join(mod_dir, f'{k}_attention.jpg'), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    elif k == 'channel':
                        with open(os.path.join(mod_dir, f'{k}_values.txt'), 'w') as f:
                            f.write(str(v[0].squeeze().cpu().numpy().tolist()))
            else:  # SimAM
                heatmap = att[0].mean(dim=0).cpu().numpy()
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
                viz = show_cam_on_image(img_rgb, heatmap, use_rgb=True)
                cv2.imwrite(os.path.join(mod_dir, 'simam_spatial_mean.jpg'), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

    return True


def run_xai(model_path, source, output_dir='xai_output', method='eigencam'):
    """Main entry point: process a single image or every image in a folder."""
    os.makedirs(output_dir, exist_ok=True)

    model, attention_modules, target_layers = setup_model(model_path)

    # Count images first for progress reporting
    images = list(image_generator(source))
    total = len(images)

    if total == 0:
        print(f"No images found in: {source}")
        return

    print(f"\n{'='*60}")
    print(f"Processing {total} image(s) with method: {method}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    success = 0
    failed = 0

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{total}] {img_path.name}")
        try:
            ok = process_single_image(img_path, model, attention_modules, target_layers, output_dir, method)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
        print()

    # Summary
    print(f"{'='*60}")
    print(f"Done! {success}/{total} succeeded, {failed} failed.")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XAI (EigenCAM / GradCAM) on YOLO predictions.")
    parser.add_argument('--model',  type=str, default='yolo11n.pt',
                        help='Path to YOLO model weights (.pt)')
    parser.add_argument('--source', type=str, default='ultralytics/assets/bus.jpg',
                        help='Path to a single image or a folder of images')
    parser.add_argument('--output', type=str, default='xai_output',
                        help='Output directory for XAI visualizations')
    parser.add_argument('--method', type=str, default='eigencam',
                        choices=['eigencam', 'gradcam', 'gradcam++', 'ss-gradcam++'],
                        help='XAI method to use')
    args = parser.parse_args()

    run_xai(args.model, args.source, args.output, args.method)
