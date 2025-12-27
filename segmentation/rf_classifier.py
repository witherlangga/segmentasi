import os
import io
import json
import joblib
import numpy as np
from PIL import Image
from rembg import remove
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_DIR = os.path.join("models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_leaf.pkl")
MAPPING_PATH = os.path.join(MODEL_DIR, "label_mapping.json")

# Ambang batas keyakinan default (dalam persen)
CONFIDENCE_THRESHOLD = float(os.environ.get('RF_CONF_THRESHOLD', 70.0))

def get_confidence_threshold():
    """Return the configured confidence threshold as a percentage."""
    return CONFIDENCE_THRESHOLD


def _ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def extract_cluster_features(image_path, K=10):
    """Return list of features: each item is (R, G, B, percentage) for every cluster."""
    with open(image_path, "rb") as f:
        input_image = f.read()

    output_image = remove(input_image)
    result = Image.open(io.BytesIO(output_image)).convert("RGBA")

    image_rgba = np.array(result)
    rgb = image_rgba[:, :, :3]
    alpha = image_rgba[:, :, 3]

    mask_leaf = alpha > 128
    pixels_leaf = np.float32(rgb[mask_leaf])
    if pixels_leaf.size == 0:
        return []

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_leaf)
    centers = np.uint8(kmeans.cluster_centers_)

    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = pixels_leaf.shape[0]

    features = []
    for i in range(len(centers)):
        pixel_count = int(counts[unique.tolist().index(i)] ) if i in unique else 0
        percentage = round((pixel_count / total_pixels) * 100, 4) if total_pixels > 0 else 0.0
        R, G, B = centers[i]
        features.append([int(R), int(G), int(B), float(percentage)])

    return features


def build_training_set(dataset_root="dataset", K=10):
    """Scan subfolders and build X,y where each cluster from an image is a sample labeled by the folder name."""
    print(f"[build_training_set] dataset_root={dataset_root}, K={K}")
    X = []
    y = []

    allowed = {"sakit", "sehat", "kurang sehat"}
    for label in os.listdir(dataset_root):
        label_path = os.path.join(dataset_root, label)
        if not os.path.isdir(label_path):
            continue

        if label.lower() not in allowed:
            # ignore unknown folders
            print(f"[build_training_set] skipping unknown folder: {label}")
            continue

        print(f"[build_training_set] scanning label: {label}")
        for fname in os.listdir(label_path):
            path = os.path.join(label_path, fname)
            if not os.path.isfile(path):
                continue

            try:
                feats = extract_cluster_features(path, K=K)
                for f in feats:
                    X.append(f)
                    y.append(label.lower())
            except Exception as e:
                # skip problematic files but log
                print(f"[build_training_set] failed processing {path}: {e}")
                continue

    print(f"[build_training_set] built samples: {len(X)}")
    return np.array(X), np.array(y)


def train_model(dataset_root="dataset", K=10, force=False):
    print(f"[train_model] start: dataset_root={dataset_root}, K={K}, force={force}")
    _ensure_model_dir()
    if os.path.exists(MODEL_PATH) and not force:
        print("[train_model] model already exists, loading")
        return load_model()

    X, y = build_training_set(dataset_root, K=K)

    print(f"[train_model] samples: X={getattr(X,'shape',None)} y={getattr(y,'shape',None)}")

    if len(X) == 0:
        print("[train_model] no training samples found, aborting")
        raise RuntimeError("Tidak ada data pelatihan (periksa folder dataset/sehat, dataset/sakit, dataset/kurang sehat)")

    # Simple encoding mapping
    labels = sorted(list(set(y)))
    mapping = {lbl: idx for idx, lbl in enumerate(labels)}
    y_encoded = np.array([mapping[v] for v in y])

    print(f"[train_model] labels: {labels}")

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)

    print(f"[train_model] training classifier: X_train={getattr(X_train,'shape',None)}")
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)

    # evaluation (printed to console)
    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, target_names=labels, zero_division=0)
    print("Random Forest training completed. Validation report:\n", report)

    # save model and mapping
    joblib.dump(clf, MODEL_PATH)
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)

    print(f"[train_model] model saved: {MODEL_PATH} mapping: {MAPPING_PATH}")
    return clf


def load_model():
    _ensure_model_dir()
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPING_PATH):
        return None

    clf = joblib.load(MODEL_PATH)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # invert mapping to get label names by index
    inv = {int(v): k for k, v in mapping.items()}
    return clf, inv


def predict_clusters(centers, percentages, clf=None):
    """centers: list of [R,G,B], percentages: list of floats (same order)"""
    model = clf or load_model()
    if model is None:
        # attempt to train automatically if model missing
        try:
            train_model()
        except Exception:
            return []
        model = load_model()

    if model is None:
        return []

    clf, inv = model

    X = []
    for c, p in zip(centers, percentages):
        X.append([int(c[0]), int(c[1]), int(c[2]), float(p)])

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    results = []
    for pred_idx, prob in zip(preds, probs):
        label = inv.get(int(pred_idx), "unknown")
        label_prob = float(max(prob))
        results.append({"label": label, "prob": round(label_prob, 4), "proba_each": [round(float(x), 4) for x in prob]})

    return results


def predict_leaf(centers, percentages, clf=None):
    model = clf or load_model()
    if model is None:
        try:
            train_model()
        except Exception:
            return {"label": "unknown", "prob": 0.0}
        model = load_model()

    if model is None:
        return {"label": "unknown", "prob": 0.0}

    clf, inv = model

    X = []
    for c, p in zip(centers, percentages):
        X.append([int(c[0]), int(c[1]), int(c[2]), float(p)])

    probs = clf.predict_proba(X)

    # Weighted aggregation by percentage
    weights = np.array(percentages, dtype=float)
    if weights.sum() <= 0:
        # fallback to equal weights
        weights = np.ones_like(weights)

    weighted = np.zeros(probs.shape[1], dtype=float)
    for w, p in zip(weights, probs):
        weighted += w * p

    # normalize
    weighted = weighted / (weights.sum() if weights.sum() > 0 else 1)

    best_idx = int(np.argmax(weighted))
    best_prob = float(weighted[best_idx])
    label = inv.get(best_idx, "unknown")

    return {"label": label, "prob": round(best_prob, 4)}
