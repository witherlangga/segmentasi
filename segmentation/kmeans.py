import os
import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from rembg import remove
from sklearn.cluster import KMeans


def _refine_leaf_mask(alpha_uint8):
    """Refine binary alpha mask (0/255) to remove noise and keep largest connected component."""
    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(alpha_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

    # keep largest connected component
    num_labels, labels = cv2.connectedComponents(m)
    if num_labels <= 1:
        return m > 0

    areas = [int((labels == i).sum()) for i in range(1, num_labels)]
    largest = 1 + int(np.argmax(areas))
    refined = (labels == largest).astype('uint8') * 255

    # final smoothing
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=2)

    return refined > 0


# KLASIFIKASI WARNA DAUN (RULE-BASED)
def classify_leaf_color(rgb_color):
    r, g, b = rgb_color

    if g > r and g > b:
        return "Daun Sehat (Hijau)"
    elif r > 150 and g > 150 and b < 100:
        return "Daun Sedikit Sakit (Kuning)"
    elif r > g and r > b and r > 120:
        return "Daun Sakit (Coklat Kekuningan)"
    elif r < 90 and g < 90 and b < 90:
        return "Daun Rusak Parah (Gelap)"
    else:
        return "Lainnya"

# SEGMENTASI DAUN DENGAN K-MEANS
def segment_leaf(image_path, K=10):

    # Remove background
    with open(image_path, "rb") as f:
        input_image = f.read()

    output_image = remove(input_image)
    result = Image.open(io.BytesIO(output_image)).convert("RGBA")

    # RGBA → RGB + Mask
    image_rgba = np.array(result)
    rgb = image_rgba[:, :, :3]
    alpha = image_rgba[:, :, 3]

    # Create initial binary mask from alpha channel
    alpha_mask = (alpha > 128).astype('uint8') * 255

    # Refine mask (morphology + keep largest component)
    try:
        refined_bool = _refine_leaf_mask(alpha_mask)
        # fallback if refinement removes too many pixels
        if np.count_nonzero(refined_bool) < 20:
            mask_leaf = alpha > 128
        else:
            mask_leaf = refined_bool
    except Exception:
        # if OpenCV fails for any reason, fallback to alpha mask
        mask_leaf = alpha > 128

    pixels_leaf = np.float32(rgb[mask_leaf])


    # K-Means
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_leaf)
    centers = np.uint8(kmeans.cluster_centers_)

    labels_keterangan = [classify_leaf_color(c) for c in centers]

    # HITUNG CLUSTER DOMINAN
    unique, counts = np.unique(labels, return_counts=True)
    dominant_index = unique[np.argmax(counts)]
    dominant_color = centers[dominant_index].tolist()
    dominant_status = classify_leaf_color(dominant_color)

    # PREDIKSI RANDOM FOREST (jika tersedia)
    try:
        from .rf_classifier import predict_clusters, predict_leaf
        # build percentages for all clusters
        total_pixels = pixels_leaf.shape[0]
        percentages_all = []
        for i in range(len(centers)):
            pixel_count_i = int(counts[unique.tolist().index(i)]) if i in unique else 0
            pcent = round((pixel_count_i / total_pixels) * 100, 4) if total_pixels > 0 else 0.0
            percentages_all.append(pcent)

        rf_clusters_pred = predict_clusters(centers, percentages_all)
        rf_leaf_pred = predict_leaf(centers, percentages_all)
    except Exception:
        rf_clusters_pred = []
        rf_leaf_pred = {"label": "unknown", "prob": 0.0}

    # GAMBAR HASIL SEGMENTASI
    segmented_image = np.zeros_like(rgb)
    segmented_image[mask_leaf] = centers[labels]

    filename = os.path.splitext(os.path.basename(image_path))[0]
    result_path = f"static/results/seg_{filename}.png"
    Image.fromarray(segmented_image).save(result_path)

    # DATA CLUSTER
    height, width, _ = rgb.shape
    label_image = np.zeros((height, width), dtype=np.uint8)
    label_image[mask_leaf] = labels

    total_pixels = pixels_leaf.shape[0]

    cluster_results = []
    numeric_data = []

    bar_labels = []
    bar_values = []

    for i, desc in enumerate(labels_keterangan):
        cluster_index = i + 1

        cluster_mask = np.zeros((height, width, 3), dtype=np.uint8)
        cluster_mask[label_image == i] = centers[i]

        cluster_path = f"static/results/cluster_{cluster_index}_{filename}.png"
        Image.fromarray(cluster_mask).save(cluster_path)

        pixel_count = np.sum(labels == i)
        percentage = round((pixel_count / total_pixels) * 100, 2)

        # RF prediction for this cluster (if available)
        rf_info = None
        try:
            rf_info = rf_clusters_pred[i]
            rf_label = rf_info.get("label", "unknown") if rf_info else "unknown"
            rf_prob = rf_info.get("prob", 0.0) if rf_info else 0.0
            rf_text = f"{rf_label.capitalize()} ({round(rf_prob*100,2)}%)"
        except Exception:
            rf_text = "-"

        cluster_results.append({
            "index": cluster_index,
            "rule_description": desc,
            "rf_text": rf_text,
            "path": cluster_path
        })

        numeric_data.append({
            "cluster": cluster_index,
            "centroid": f"({centers[i][0]}, {centers[i][1]}, {centers[i][2]})",
            "pixel_count": int(pixel_count),
            "percentage": percentage,
            "rule_description": desc,
            "rf_prediction": rf_info.get("label") if rf_info else None,
            "rf_prob": rf_info.get("prob") if rf_info else None
        })

        bar_labels.append(f"Cluster {cluster_index}")
        bar_values.append(percentage)

    # GRAFIK BAR
    bar_path = f"static/results/bar_{filename}.png"

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bar_labels, bar_values)

    plt.title("Distribusi Persentase Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Persentase (%)")

    for bar, value in zip(bars, bar_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()

    # KESIMPULAN PER-DAUN (AGREGASI RF)
    try:
        from .rf_classifier import get_confidence_threshold

        rf_leaf_label = rf_leaf_pred.get("label", "unknown").capitalize()
        rf_leaf_prob = round(rf_leaf_pred.get("prob", 0.0) * 100, 2)

        threshold = get_confidence_threshold()

        # Tentukan level kepercayaan berdasarkan nilai probabilitas
        if rf_leaf_prob >= threshold:
            confidence_level = "Tinggi"
        elif rf_leaf_prob >= 50:
            confidence_level = "Sedang"
        else:
            confidence_level = "Rendah"

        # Buat kesimpulan yang lebih detail dan profesional (Bahasa Indonesia)
        conclusion = (
            f"Kesimpulan:\n"
            f"- Prediksi model: {rf_leaf_label} (tingkat keyakinan: {rf_leaf_prob}%).\n"
            f"- Status keyakinan: {confidence_level} (ambang: {threshold}%).\n"
            f"- Analisis rule-based (warna) menunjukkan klaster dominan: {dominant_status}.\n\n"
            f"Interpretasi:\n"
            f"Model Random Forest memberikan indikasi bahwa kondisi daun cenderung {rf_leaf_label.lower()}. "
            f"Tingkat keyakinan model sebesar {rf_leaf_prob}% menunjukkan {('tingkat kepercayaan yang cukup baik.' if rf_leaf_prob >= threshold else 'hasil yang perlu verifikasi lebih lanjut.')}\n\n"
            f"Rekomendasi:\n"
            f"- Jika probabilitas >= {threshold}%: hasil dapat dijadikan dasar pengambilan keputusan awal.\n"
            f"- Jika probabilitas < {threshold}%: lakukan pemeriksaan visual tambahan atau kumpulkan lebih banyak sampel dan ulangi analisis.\n"
            f"- Gabungkan hasil ini dengan konteks lapangan (mis. gejala lain, kondisi lingkungan) sebelum mengambil tindakan.\n"
        )
    except Exception:
        rf_leaf_label = "unknown"
        rf_leaf_prob = 0.0
        confidence_level = "Tidak diketahui"
        threshold = get_confidence_threshold() if 'get_confidence_threshold' in globals() else 70.0
        conclusion = "Tidak ada prediksi RF tersedia."

    # Short, single-line conclusion (only what is shown to users)
    short_conclusion = f"Prediksi: {rf_leaf_label} ({rf_leaf_prob}%). Dominan: {dominant_status}."

    # RETURN KE FLASK
    return {
        "result_path": result_path,
        "clusters": cluster_results,
        "numeric_data": numeric_data,
        "bar_chart": bar_path,
        "dominant_color": dominant_color,
        "dominant_status": dominant_status,
        "rf_clusters": rf_clusters_pred,
        "rf_leaf_prediction": {"label": rf_leaf_label, "prob": rf_leaf_prob},
        "confidence_level": confidence_level,
        "confidence_threshold": threshold,
        "conclusion": conclusion,
        "short_conclusion": short_conclusion
    }
