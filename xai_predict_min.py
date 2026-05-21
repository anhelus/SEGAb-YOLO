import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

class DirectActivationXAI:
    def __init__(self, model_path, layer_type="GAM"):
        """
        layer_type può essere: "GAM", "SIMAM" (modelli custom) o "BASELINE" (modello standard)
        """
        self.model = YOLO(model_path)
        self.layer_type = layer_type.upper()
        self.activation = None
        self._register_hooks()

    def _forward_hook(self, module, input, output):
        self.activation = output.detach()

    def _register_hooks(self):
        found = False
        inner_model = self.model.model.model
        
        for name, module in inner_model.named_modules():
            class_name = type(module).__name__
            
            # Controllo dinamico sul tipo di layer richiesto
            if self.layer_type in ("GAM", "SIMAM") and class_name == self.layer_type:
                module.register_forward_hook(self._forward_hook)
                print(f"-> Hook XAI registrato con successo sul modulo Custom: {name} ({class_name})")
                found = True
                break
            elif self.layer_type == "BASELINE" and "SPPF" in class_name:
                target = module.cv2 if hasattr(module, 'cv2') else module
                target.register_forward_hook(self._forward_hook)
                print(f"-> Hook XAI registrato con successo sulla Baseline: {name} (SPPF Out)")
                found = True
                break
                
        if not found:
            raise RuntimeError(f"Impossibile trovare un layer corrispondente a: {self.layer_type}. Controlla i pesi e lo YAML.")

    def generate_heatmap(self, img_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        img = cv2.imread(img_path)
        h_orig, w_orig, _ = img.shape

        results = self.model(img, verbose=False)[0]
        
        if self.activation is None:
            raise ValueError("Errore: L'hook non ha intercettato alcun tensore.")

        feature_map = self.activation[0] 
        channels = feature_map.shape[0]
        
        heatmap = torch.sqrt(torch.sum(feature_map ** 2, dim=0)).cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)

        heatmap_resized = cv2.resize(heatmap, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{results.names[cls]} {conf:.2f}"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 0, 180), 3)
            cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 0, 180), 2)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_activation_{self.layer_type.lower()}.jpg")
        cv2.imwrite(out_path, overlay)
        print(f"Salvata mappa in: {out_path} (Canali: {channels})")

# --- ESECUZIONE COMPLETA TRILATERALE ---
if __name__ == "__main__":
    IMG_TEST = "percorso/all_immagine.jpg"
    
    # Esegui il test sui tre modelli salvando tutto in una cartella pulita
    models_config = {
        "BASELINE": "percorso/modello_base.pt",
        "GAM": "percorso/modello_gam.pt",
        "SIMAM": "percorso/modello_simam.pt"
    }
    
    for layer, path in models_config.items():
        print(f"\nElaborazione configurazione: {layer}")
        try:
            xai = DirectActivationXAI(path, layer_type=layer)
            xai.generate_heatmap(IMG_TEST, output_dir="results/xai_comparison")
        except Exception as e:
            print(f"Errore su {layer}: {e}")