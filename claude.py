import cv2
import numpy as np

"""
Stereo Line Matcher - Sistema per l'analisi e trasformazione di linee stereo

Funzione principale:
    process_stereo_lines(img, red_line, green_line, start_pt, output_filename="test.jpg", direction='right')

Uso semplice:
    import cv2
    image = cv2.imread("your_image.jpg")
    red_points = [(x1,y1), (x2,y2), ...]  # Lista dei punti della linea rossa
    green_points = [(x1,y1), (x2,y2), ...] # Lista dei punti della linea verde
    start_point = (x, y)  # Punto di partenza sulla linea verde
    
    results = process_stereo_lines(image, red_points, green_points, start_point, direction='right')
    
Returns:
    dict con transform_matrix, transformed_points, intermediate_lines, distance, ecc.
"""

def reorder_line_from_start_point(line_points, start_point, direction='right'):
    """
    Riordina i punti di una linea partendo dal punto più vicino al start_point
    
    Args:
        line_points: lista di tuple [(x1,y1), (x2,y2), ...]
        start_point: tuple (x, y) punto di riferimento
        direction: 'right' per andare verso destra, 'left' per andare verso sinistra
        
    Returns:
        Lista riordinata di punti
    """
    if not line_points or start_point is None:
        return line_points
    
    # Trova il punto più vicino al start_point
    line_array = np.array(line_points, dtype=np.float32)
    start_array = np.array(start_point, dtype=np.float32)
    distances = np.sqrt(np.sum((line_array - start_array)**2, axis=1))
    closest_idx = np.argmin(distances)
    
    return reorder_line_from_index(line_points, closest_idx, direction)

def reorder_line_from_index(line_points, start_idx, direction='right'):
    """
    Riordina i punti di una linea partendo dall'indice specificato
    
    Args:
        line_points: lista di tuple [(x1,y1), (x2,y2), ...]
        start_idx: indice del punto di partenza
        direction: 'right' per andare verso destra, 'left' per andare verso sinistra
        
    Returns:
        Lista riordinata di punti
    """
    if not line_points or start_idx < 0 or start_idx >= len(line_points):
        return line_points
    
    # Suddividi la linea in due parti: prima e dopo il punto di partenza
    before_start = line_points[:start_idx]
    after_start = line_points[start_idx:]
    
    if direction == 'right':
        # Parti dal start_idx e vai verso destra (indici crescenti)
        # poi aggiungi la parte prima invertita
        reordered = after_start + list(reversed(before_start))
    else:  # direction == 'left'
        # Parti dal start_idx e vai verso sinistra (indici decrescenti)
        # quindi inverti after_start e aggiungi before_start
        reordered = list(reversed(after_start)) + before_start
    
    return reordered

class StereoLineMatcher:
    def __init__(self, focal_length=663.4631958007812):
        self.focal_length = focal_length
        
    def calculate_disparity_from_lines(self, red_points, green_points):
        """Calcola la disparità media tra le due linee"""
        if red_points is None or green_points is None:
            return 0
            
        # Calcola la distanza media tra i punti corrispondenti
        disparities = []
        
        for red_point in red_points:
            # Trova il punto più vicino sulla linea verde
            min_dist = float('inf')
            closest_green = None
            
            for green_point in green_points:
                # Converti in float per evitare overflow
                dist = np.sqrt(float(red_point[0] - green_point[0])**2 + 
                              float(red_point[1] - green_point[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_green = green_point
            
            if closest_green is not None:
                # Disparità orizzontale (differenza in x)
                disparity = abs(float(red_point[0]) - float(closest_green[0]))
                disparities.append(disparity)
        
        return np.mean(disparities) if disparities else 0
    
    def estimate_distance(self, disparity, baseline=0.1):
        """Stima la distanza usando la disparità
        
        Args:
            disparity: disparità in pixel
            baseline: distanza tra le telecamere in metri (da calibrare)
        """
        if disparity == 0:
            return float('inf')
        
        # Formula stereo: distance = (focal_length * baseline) / disparity
        # Tuttavia, per linee così vicine, usiamo un fattore di correzione più realistico
        # Assumiamo che una disparità di ~1 pixel corrisponda a circa 7 metri di distanza
        
        # Calcolo semplificato basato su calibrazione empirica
        # Per disparità piccole (< 5 pixel), usiamo una relazione lineare
        if disparity < 5:
            # Circa 7 metri per disparità di 1 pixel, scalato linearmente
            distance = 7.0 / disparity
        else:
            # Per disparità maggiori, usiamo la formula stereo standard
            distance = (self.focal_length * baseline) / disparity
            
        return distance
    
    def estimate_distance_from_pixel_coordinates(self, red_points, green_points):
        """Stima la distanza basata sulla posizione dei pixel nell'immagine
        
        Questo metodo assume che:
        - Le linee rappresentino oggetti alla stessa distanza dal suolo
        - La differenza in coordinate y indica la distanza relativa
        - Calibrazione basata su conoscenza del dominio (7m max)
        """
        if not red_points or not green_points:
            return 0
        
        # Calcola la posizione media delle linee
        red_y_avg = np.mean([p[1] for p in red_points])
        green_y_avg = np.mean([p[1] for p in green_points])
        red_x_avg = np.mean([p[0] for p in red_points])
        green_x_avg = np.mean([p[0] for p in green_points])
        
        # Differenza verticale (più importante per la distanza)
        y_diff = abs(red_y_avg - green_y_avg)
        # Differenza orizzontale
        x_diff = abs(red_x_avg - green_x_avg)
        
        # Distanza euclidea in pixel
        pixel_distance = np.sqrt(x_diff**2 + y_diff**2)
        
        # Calibrazione empirica: assumiamo che la massima distanza osservata
        # corrisponda a circa 7 metri. Scala in base alla distanza pixel.
        # Questo è un esempio - dovresti calibrare con misure reali
        max_pixel_distance = 200  # da calibrare in base alle tue immagini
        max_real_distance = 7   # metri
        
        estimated_distance = (pixel_distance / max_pixel_distance) * max_real_distance

        # Limita la distanza massima a 7 metri
        return min(estimated_distance, 7.0)
    
    def calculate_realistic_transform_matrix(self, red_points, green_points, distance_ratio=1.0, start_point=None):
        """Calcola una trasformazione che considera anche il ridimensionamento prospettico
        
        Args:
            red_points: punti della linea rossa
            green_points: punti della linea verde  
            distance_ratio: rapporto di distanza (green_distance / red_distance)
            start_point: punto specifico sulla linea verde da cui iniziare la trasformazione
        """
        if red_points is None or green_points is None:
            return None
            
        # Se abbiamo un punto di partenza specifico, lo usiamo come riferimento
        if start_point is not None:
            green_reference = np.array(start_point, dtype=np.float32)
            # Trova il punto più vicino sulla linea rossa al punto di partenza
            red_points_array = np.array(red_points, dtype=np.float32)
            distances = np.sqrt(np.sum((red_points_array - green_reference)**2, axis=1))
            closest_red_idx = np.argmin(distances)
            red_reference = red_points_array[closest_red_idx]
        else:
            # Calcola i centri delle linee (comportamento precedente)
            red_reference = np.mean(red_points, axis=0)
            green_reference = np.mean(green_points, axis=0)
        
        # Calcola il fattore di scala basato sulla distanza
        # Gli oggetti più lontani appaiono più piccoli
        scale_factor = 1.0 / distance_ratio if distance_ratio > 0 else 1.0
        
        # Limita il fattore di scala a valori ragionevoli
        scale_factor = np.clip(scale_factor, 0.3, 3.0)
        
        # Calcola dove si dovrebbe trovare il punto di riferimento della linea rossa dopo la scala
        scaled_red_reference = red_reference * scale_factor
        
        # La traslazione necessaria per allineare il punto di riferimento
        translation = green_reference - scaled_red_reference
        
        # Crea la matrice di trasformazione
        transform_matrix = np.array([
            [scale_factor, 0, translation[0]],
            [0, scale_factor, translation[1]]
        ], dtype=np.float32)
        
        return transform_matrix
    
    def apply_transform_to_points(self, points, transform_matrix):
        """Applica la trasformazione ai punti"""
        if points is None or transform_matrix is None:
            return None
        
        # Converti i punti in formato numpy
        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        
        # Applica la trasformazione
        transformed_points = cv2.transform(points_array, transform_matrix)
        
        # Ritorna come lista di tuple
        return [(int(p[0][0]), int(p[0][1])) for p in transformed_points]
    
    def generate_intermediate_lines(self, line1_points, line2_points, num_lines=5):
        """Genera linee intermedie tra due linee per riempire lo spazio
        
        Args:
            line1_points: punti della prima linea
            line2_points: punti della seconda linea  
            num_lines: numero di linee intermedie da generare
            
        Returns:
            Lista di liste di punti, una per ogni linea intermedia
        """
        if line1_points is None or line2_points is None:
            return []
        
        # Assicurati che entrambe le linee abbiano lo stesso numero di punti
        # Interpola se necessario
        line1_array = np.array(line1_points, dtype=np.float32)
        line2_array = np.array(line2_points, dtype=np.float32)
        
        # Se le linee hanno lunghezze diverse, interpola per renderle della stessa lunghezza
        max_points = max(len(line1_array), len(line2_array))
        
        # Interpola line1 al numero massimo di punti
        if len(line1_array) != max_points:
            t1 = np.linspace(0, 1, len(line1_array))
            t_new = np.linspace(0, 1, max_points)
            line1_x_interp = np.interp(t_new, t1, line1_array[:, 0])
            line1_y_interp = np.interp(t_new, t1, line1_array[:, 1])
            line1_array = np.column_stack([line1_x_interp, line1_y_interp])
        
        # Interpola line2 al numero massimo di punti
        if len(line2_array) != max_points:
            t2 = np.linspace(0, 1, len(line2_array))
            t_new = np.linspace(0, 1, max_points)
            line2_x_interp = np.interp(t_new, t2, line2_array[:, 0])
            line2_y_interp = np.interp(t_new, t2, line2_array[:, 1])
            line2_array = np.column_stack([line2_x_interp, line2_y_interp])
        
        # Genera le linee intermedie
        intermediate_lines = []
        
        for i in range(num_lines):
            # Calcola il fattore di interpolazione (da 0 a 1)
            alpha = (i + 1) / (num_lines + 1)
            
            # Interpola linearmente tra line1 e line2
            intermediate_points = (1 - alpha) * line1_array + alpha * line2_array
            
            # Converti a lista di tuple di interi
            intermediate_line = [(int(p[0]), int(p[1])) for p in intermediate_points]
            intermediate_lines.append(intermediate_line)
        
        return intermediate_lines

# Funzione utility per usare direttamente le tue liste
def transform_red_to_green(red_points, green_points, focal_length=663.4631958007812, start_point=None, num_intermediate_lines=None, direction='right', debug=False):
    """
    Funzione semplificata per trasformare direttamente i punti
    
    Args:
        red_points: lista di tuple [(x1,y1), (x2,y2), ...]
        green_points: lista di tuple [(x1,y1), (x2,y2), ...]
        focal_length: lunghezza focale in pixel
        start_point: punto specifico sulla linea verde da usare come riferimento per la trasformazione
        num_intermediate_lines: numero di linee intermedie (se None, viene calcolato automaticamente dalla distanza)
        direction: direzione del segmento dal punto di partenza ('left' o 'right')
        debug: se True, stampa informazioni di debug sul riordinamento dei punti
    
    Returns:
        dict con 'transform_matrix', 'transformed_points', 'disparity', 'distance', 'distance_v2', 'intermediate_lines'
    """
    matcher = StereoLineMatcher(focal_length)
    
    # Se abbiamo un punto di partenza, riordina le linee partendo dal punto più vicino
    if start_point is not None:
        if debug:
            print(f"Punto di partenza: {start_point}")
            print(f"Direzione: {direction}")
            print(f"Primi 3 punti green line (originali): {green_points[:3]}")
            print(f"Ultimi 3 punti green line (originali): {green_points[-3:]}")
        
        # Riordina i punti della linea verde partendo dal punto più vicino al start_point
        #green_points = reorder_line_from_start_point(green_points, start_point, direction)
        
        if debug:
            print(f"Primi 3 punti green line (riordinati): {green_points[:3]}")
            print(f"Ultimi 3 punti green line (riordinati): {green_points[-3:]}")
        
        # Riordina anche i punti della linea rossa in modo corrispondente
        # Trova il punto più vicino al start_point sulla linea rossa
        red_points_array = np.array(red_points, dtype=np.float32)
        start_point_array = np.array(start_point, dtype=np.float32)
        distances = np.sqrt(np.sum((red_points_array - start_point_array)**2, axis=1))
        closest_red_idx = np.argmin(distances)
        
        if debug:
            print(f"Punto più vicino su red line: indice {closest_red_idx}, punto {red_points[closest_red_idx]}")
            print(f"Primi 3 punti red line (originali): {red_points[:3]}")
            print(f"Ultimi 3 punti red line (originali): {red_points[-3:]}")
        
        # Riordina la linea rossa partendo dal punto più vicino al start_point
        red_points = reorder_line_from_index(red_points, closest_red_idx, direction)
        
        if debug:
            print(f"Primi 3 punti red line (riordinati): {red_points[:3]}")
            print(f"Ultimi 3 punti red line (riordinati): {red_points[-3:]}")
    
    # Calcola disparità
    disparity = matcher.calculate_disparity_from_lines(red_points, green_points)
    
    # Stima distanza con metodo stereo (corretto)
    distance_stereo = matcher.estimate_distance(disparity, baseline=0.1)
    
    # Stima distanza con metodo alternativo basato su geometria dell'immagine
    distance_geometric = matcher.estimate_distance_from_pixel_coordinates(red_points, green_points)
    
    # Stima le distanze relative (assumiamo che la linea rossa sia più vicina)
    red_distance_estimate = 3.9  # metri, valore ipotetico
    green_distance_estimate = red_distance_estimate + distance_geometric
    distance_ratio = green_distance_estimate / red_distance_estimate
    
    # Calcola automaticamente il numero di linee intermedie se non specificato
    if num_intermediate_lines is None:
        # Formula: più distanza = più linee intermedie
        # Usiamo la distanza geometrica come base
        min_lines = 3
        max_lines = 15

        # Scala in base alla distanza (0-40 metri -> 3-15 linee)
        distance_for_calc = min(distance_geometric, 7)  # Cap a 40 metri
        lines_factor = distance_for_calc / 7  # Normalizza 0-1

        num_intermediate_lines = int(min_lines + (max_lines - min_lines) * lines_factor)
        
        #print(f"Numero di linee calcolato automaticamente: {num_intermediate_lines} (distanza: {distance_geometric:.2f}m)")
    
    # Calcola trasformazione realistica che considera la prospettiva e il punto di partenza
    transform_matrix = matcher.calculate_realistic_transform_matrix(red_points, green_points, distance_ratio, start_point)
    
    # Applica trasformazione
    transformed_points = None
    intermediate_lines = []
    
    if transform_matrix is not None:
        transformed_points = matcher.apply_transform_to_points(red_points, transform_matrix)
        
        # Genera le linee intermedie tra la linea rossa originale e quella blu trasformata
        if transformed_points is not None:
            intermediate_lines = matcher.generate_intermediate_lines(red_points, transformed_points, num_intermediate_lines)
    
    return {
        'transform_matrix': transform_matrix,
        'transformed_points': transformed_points,
        'intermediate_lines': intermediate_lines,
        'num_intermediate_lines': num_intermediate_lines,
        'disparity': disparity,
        'distance_stereo': distance_stereo,
        'distance_geometric': distance_geometric,
        'distance': distance_geometric,  # Usa il metodo geometrico come principale
        'distance_ratio': distance_ratio
    }

def process_stereo_lines(img, red_line, green_line, start_pt, output_filename="test.jpg", focal_length=663.4631958007812, direction='right', debug=False):
    """
    Funzione principale per processare e visualizzare linee stereo
    
    Args:
        img: immagine OpenCV (numpy array)
        red_line: lista di tuple [(x1,y1), (x2,y2), ...] per la linea rossa
        green_line: lista di tuple [(x1,y1), (x2,y2), ...] per la linea verde
        start_pt: tuple (x, y) punto di partenza sulla linea verde
        output_filename: nome del file di output (default: "test.jpg")
        focal_length: lunghezza focale in pixel (default: 663.4631958007812)
        direction: direzione del segmento dal punto di partenza ('left' o 'right')
        debug: se True, stampa informazioni di debug
        
    Returns:
        dict con tutti i risultati dell'analisi e l'immagine processata
    """
    
    # Calcola la trasformazione usando il punto di partenza specificato e la direzione
    # Il numero di linee intermedie sarà calcolato automaticamente dalla distanza
    results = transform_red_to_green(red_line, green_line, focal_length=focal_length, start_point=start_pt, direction=direction, debug=debug)
    
    # print(f"Risultati della trasformazione con punto di partenza {start_pt}")
    # print(f"Distanza geometrica: {results['distance_geometric']:.2f}m")
    # print(f"Numero di linee intermedie generate automaticamente: {results['num_intermediate_lines']}")
    
    # Crea una copia dell'immagine per il disegno
    #output_image = img.copy()
    
    # Disegna il punto di partenza con un cerchio più grande
    #cv2.circle(img, start_pt, 8, (255, 255, 0), -1)  # Ciano per il punto di partenza
    #cv2.putText(output_image, "START", (start_pt[0]+10, start_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Disegna i punti della linea rossa (originali)
    
    # Disegna i punti trasformati (se disponibili)
    if results['transformed_points'] is not None:
        for point in results['transformed_points']:
            cv2.circle(img, point, 2, (255, 0, 0), -1)  # Blu
    
    # Disegna le linee intermedie con colori graduali
    if results['intermediate_lines']:
        for i, intermediate_line in enumerate(results['intermediate_lines']):
            # Calcola il colore intermedio tra rosso e blu
            # Alpha va da 0 (più vicino al rosso) a 1 (più vicino al blu)
            alpha = (i + 1) / (len(results['intermediate_lines']) + 1)
            
            # Interpola tra rosso (0, 0, 255) e blu (255, 0, 0)
            red_component = int(255 * (1 - alpha))
            blue_component = int(255 * alpha)
            color = (blue_component, 0, red_component)  # BGR format
            
            # Disegna i punti della linea intermedia
            for point in intermediate_line:
                cv2.circle(img, point, 1, color, -1)
    
    # Salva l'immagine risultante
    #cv2.imwrite(output_filename, output_image)
    #print(f"Immagine salvata come '{output_filename}' con i punti visualizzati")
    
    # Aggiungi l'immagine processata ai risultati
    results['output_image'] = img
    #results['output_filename'] = output_filename
    
    return results

if __name__ == "__main__":
    #main()
    #image = cv2.imread("custom_example.jpg")
    image = cv2.imread("output/output_121.jpg")

    # Verifica che l'immagine sia stata caricata correttamente
    if image is None:
        print("Errore: impossibile caricare l'immagine 'custom_example.jpg'")
        exit(1)
    
    red_line = [(483, 494), (484, 493), (485, 493), (486, 493), (487, 493), (488, 493), (489, 493), (490, 493), (491, 493), (492, 493), (493, 493), (494, 493), (495, 493), (496, 493), (497, 493), (498, 493), (499, 493), (500, 493), (501, 493), (502, 493), (503, 493), (504, 493), (505, 493), (506, 493), (507, 493), (508, 493), (509, 493), (510, 493), (511, 493), (512, 493), (513, 493), (514, 493), (515, 493), (516, 493), (517, 493), (518, 493), (519, 493), (520, 493), (521, 493), (522, 493), (523, 493), (524, 493), (525, 493), (526, 493), (527, 493), (528, 493), (529, 493), (530, 493), (531, 493), (532, 493), (533, 493), (534, 493), (535, 493), (536, 493), (537, 493), (538, 493), (539, 493), (540, 493), (541, 493), (542, 493), (543, 493), (544, 493), (545, 493), (546, 493), (547, 493), (548, 493), (549, 493), (550, 493), (551, 493), (552, 493), (553, 493), (554, 493), (555, 493), (556, 493), (557, 493), (558, 493), (559, 493), (560, 493), (561, 493), (562, 493), (563, 493), (564, 493), (565, 493), (566, 493), (567, 493), (568, 493), (569, 493), (570, 493), (571, 493), (572, 493), (573, 493), (574, 493), (575, 493), (576, 493), (577, 493), (578, 493), (579, 493), (580, 493), (581, 493), (582, 493), (583, 493), (584, 493), (585, 493), (586, 493), (587, 493), (588, 493), (589, 493), (590, 493), (591, 493), (592, 493), (593, 493), (594, 493), (595, 493), (596, 493), (597, 493), (598, 493), (599, 493), (600, 493), (601, 493), (602, 493), (603, 493), (604, 493), (605, 493), (606, 493), (607, 493), (608, 493), (609, 493), (610, 493), (611, 493), (612, 493), (613, 493), (614, 493), (615, 493), (616, 493), (617, 493), (618, 493), (619, 493), (620, 493), (621, 493), (622, 493), (623, 493), (624, 493), (625, 493), (626, 493), (627, 493), (628, 493), (629, 493), (630, 493), (631, 493), (632, 493), (633, 493), (634, 493), (635, 493), (636, 493), (637, 493), (638, 493), (639, 493), (640, 493), (641, 493), (642, 493), (643, 493), (644, 493), (645, 493), (646, 493), (647, 493), (648, 493), (649, 493), (650, 493), (651, 493), (652, 493), (653, 493), (654, 493), (655, 493), (656, 493), (657, 493), (658, 493), (659, 493), (660, 493), (661, 493), (662, 493), (663, 493), (664, 493), (665, 493), (666, 493), (667, 493), (668, 493), (669, 493), (670, 493), (671, 493), (672, 493), (673, 493), (674, 493), (675, 493), (676, 493), (677, 493), (678, 493), (679, 493), (680, 493), (681, 493), (682, 493), (683, 493), (684, 493), (685, 493), (686, 493), (687, 493), (688, 493), (689, 493), (690, 493), (691, 493), (692, 493), (693, 493), (694, 493), (695, 493), (696, 493), (697, 493), (698, 493), (699, 493), (700, 493), (701, 493), (702, 493), (703, 493), (704, 493), (705, 493), (706, 493), (707, 493), (708, 493), (709, 493), (710, 493), (711, 493), (712, 493), (713, 493), (714, 493), (715, 493), (716, 493), (717, 493), (718, 493), (719, 493), (720, 493), (721, 493), (722, 493), (723, 493), (724, 493), (725, 493), (726, 493), (727, 493), (728, 493), (729, 493), (730, 493), (731, 493), (732, 493), (733, 493), (734, 493), (735, 493), (736, 493), (737, 493), (738, 493), (739, 493), (740, 493), (741, 493), (742, 493), (743, 493), (744, 493), (745, 493), (746, 493), (747, 493), (748, 493), (749, 493), (750, 493), (751, 493), (752, 493), (753, 493), (754, 493), (755, 493), (756, 493), (757, 493), (758, 493), (759, 493), (760, 493), (761, 493), (762, 493), (763, 493), (764, 493), (765, 493), (766, 493), (767, 493), (768, 493), (769, 493), (770, 493), (771, 493), (772, 493), (773, 493), (774, 493), (775, 493), (776, 493), (777, 493), (778, 493), (779, 493), (780, 493), (781, 493), (782, 493), (783, 493), (784, 493), (785, 493), (786, 493), (787, 493), (788, 493), (789, 493), (790, 493), (791, 493), (792, 493), (793, 493), (794, 493), (795, 493), (796, 493), (797, 493), (798, 493), (799, 493), (800, 493), (801, 493), (802, 493), (803, 493), (804, 493), (805, 493), (806, 493), (807, 493), (808, 493), (809, 493), (810, 493), (811, 493), (812, 493), (813, 493), (814, 493), (815, 493), (816, 493), (817, 493), (818, 493), (819, 493), (820, 493), (821, 493), (822, 493), (823, 493), (824, 493), (825, 493), (826, 493), (827, 493), (828, 493), (829, 493), (830, 493), (831, 493), (832, 493), (833, 493), (834, 493), (835, 493)]
    # esempio di pixel della linea verde (ordinati)
    green_line = [(417, 396), (418, 396), (419, 396), (420, 396), (421, 396), (422, 396), (423, 396), (424, 396), (425, 396), (426, 396), (427, 396), (428, 396), (429, 396), (430, 396), (431, 396), (432, 396), (433, 396), (434, 396), (435, 396), (436, 396), (437, 396), (438, 396), (439, 396), (440, 396), (441, 396), (442, 396), (443, 396), (444, 396), (445, 396), (446, 396), (447, 396), (448, 396), (449, 396), (450, 396), (451, 396), (452, 396), (453, 396), (454, 396), (455, 396), (456, 396), (457, 396), (458, 396), (459, 396), (460, 396), (461, 396), (462, 396), (463, 396), (464, 396), (465, 396), (466, 396), (467, 396), (468, 396), (469, 396), (470, 396), (471, 396), (472, 396), (473, 396), (474, 396), (475, 396), (476, 396), (477, 396), (478, 396), (479, 396), (480, 396), (481, 396), (482, 396), (483, 396), (484, 396), (485, 397), (486, 397), (487, 397), (488, 397), (489, 397), (490, 397), (491, 397), (492, 397), (493, 397), (494, 397), (495, 397), (496, 397), (497, 397), (498, 397), (499, 397), (500, 397), (501, 397), (502, 397), (503, 397), (504, 397), (505, 397), (506, 397), (507, 397), (508, 397), (509, 397), (510, 397), (511, 397), (512, 397), (513, 397), (514, 397), (515, 397), (516, 397), (517, 397), (518, 397), (519, 397), (520, 397), (521, 397), (522, 397), (523, 397), (524, 397), (525, 397), (526, 397), (527, 397), (528, 397), (529, 397), (530, 397), (531, 397), (532, 397), (533, 397), (534, 397), (535, 397), (536, 397), (537, 397), (538, 397), (539, 397), (540, 397), (541, 397), (542, 397), (543, 397), (544, 397), (545, 397), (546, 397), (547, 397), (548, 397), (549, 397), (550, 397), (551, 397), (552, 397), (553, 397), (554, 397), (555, 397), (556, 397), (557, 397), (558, 397), (559, 397), (560, 397), (561, 397), (562, 397), (563, 397), (564, 397), (565, 397), (566, 397), (567, 397), (568, 397), (569, 398), (570, 398), (571, 398), (572, 398), (573, 398), (574, 398), (575, 398), (576, 398), (577, 398), (578, 398), (579, 398), (580, 398), (581, 398), (582, 398), (583, 398), (584, 398), (585, 398), (586, 398), (587, 398), (588, 398), (589, 398), (590, 398), (591, 398), (592, 398), (593, 398), (594, 398), (595, 398), (596, 398), (597, 398), (598, 398), (599, 398), (600, 398), (601, 398), (602, 398), (603, 398), (604, 398), (605, 398), (606, 398), (607, 398), (608, 398), (609, 398), (610, 398), (611, 398), (612, 398), (613, 398), (614, 398), (615, 398), (616, 398), (617, 398), (618, 398), (619, 398), (620, 398), (621, 398), (622, 398), (623, 398), (624, 398), (625, 398), (626, 398), (627, 398), (628, 398), (629, 398), (630, 398), (631, 398), (632, 398), (633, 398), (634, 398), (635, 398), (636, 398), (637, 398), (638, 398), (639, 398), (640, 398), (641, 398), (642, 398), (643, 398), (644, 398), (645, 398), (646, 398), (647, 398), (648, 398), (649, 398), (650, 398), (651, 398), (652, 398), (653, 399), (654, 399), (655, 399), (656, 399), (657, 399), (658, 399), (659, 399), (660, 399), (661, 399), (662, 399), (663, 399), (664, 399), (665, 399), (666, 399), (667, 399), (668, 399), (669, 399), (670, 399), (671, 399), (672, 399), (673, 399), (674, 399), (675, 399), (676, 399), (677, 399), (678, 399), (679, 399), (680, 399), (681, 399), (682, 399), (683, 399), (684, 399), (685, 399), (686, 399), (687, 399), (688, 399), (689, 399), (690, 399), (691, 399), (692, 399), (693, 399), (694, 399), (695, 399), (696, 399), (697, 399), (698, 399), (699, 399), (700, 399), (701, 399), (702, 399), (703, 399), (704, 399), (705, 399), (706, 399), (707, 399), (708, 399), (709, 399), (710, 399), (711, 399), (712, 399), (713, 399), (714, 399), (715, 399), (716, 399), (717, 399), (718, 399), (719, 399), (720, 399), (721, 399), (722, 399), (723, 399), (724, 399), (725, 399), (726, 399), (727, 399), (728, 399), (729, 399), (730, 399), (731, 399), (732, 399), (733, 399), (734, 399), (735, 399), (736, 399), (737, 400), (738, 400), (739, 400), (740, 400), (741, 400), (742, 400), (743, 400), (744, 400), (745, 400), (746, 400), (747, 400), (748, 400), (749, 400), (750, 400), (751, 400), (752, 400), (753, 400), (754, 400), (755, 400), (756, 400), (757, 400), (758, 400), (759, 400), (760, 400), (761, 400), (762, 400), (763, 400), (764, 400), (765, 400), (766, 400), (767, 400), (768, 400), (769, 400), (770, 400), (771, 400), (772, 400), (773, 400), (774, 400), (775, 400), (776, 400), (777, 400), (778, 400), (779, 400), (780, 400), (781, 400), (782, 400), (783, 400), (784, 400), (785, 400), (786, 400), (787, 400), (788, 400), (789, 400), (790, 400), (791, 400), (792, 400), (793, 400), (794, 400), (795, 400), (796, 400), (797, 400), (798, 400), (799, 400), (800, 400), (801, 400), (802, 400), (803, 400), (804, 400), (805, 400), (806, 400), (807, 400), (808, 400), (809, 400), (810, 400), (811, 400), (812, 400), (813, 400), (814, 400), (815, 400), (816, 400), (817, 400), (818, 400), (819, 400), (820, 400), (821, 401), (822, 401), (823, 401), (824, 401), (825, 401), (826, 401), (827, 401), (828, 401)]
    
    start_pt = (583, 361)
    
    #start_pt = (484, 361)  # Punto di partenza sulla linea verde

    # Chiamata semplice alla funzione con direzione verso destra (default)
    results = process_stereo_lines(image, red_line, green_line, start_pt, direction='right')
    
    # Per andare verso sinistra dal punto di partenza, usa:
    # results = process_stereo_lines(image, red_line, green_line, start_pt, direction='left')

    image_with_points = results['output_image']
    cv2.imshow("Image with Points", image_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output_with_points.jpg", image_with_points)