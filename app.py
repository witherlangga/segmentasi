import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from segmentation.kmeans import segment_leaf
# Random Forest helper (will train/load model from dataset folder)
from segmentation.rf_classifier import load_model, train_model

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Pastikan folder ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Bersihkan "static/" dari path hasil kmeans
def clean_static_path(path):
    if path.startswith("static/"):
        return path.replace("static/", "", 1)
    return path

# ROUTE UPLOAD GAMBAR (MULTIPLE)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        # CEK INPUT FILE
        if 'images[]' not in request.files:
            return redirect(request.url)

        files = request.files.getlist('images[]')
        k = int(request.form.get('k', 10))

        # Pastikan model RF tersedia (latih jika belum ada)
        try:
            if load_model() is None:
                train_model()
        except Exception:
            # Jika training gagal, lanjutkan dengan rule-based
            pass

        if len(files) == 0:
            return redirect(request.url)

        all_results = []

        # LOOP SEMUA GAMBAR
        for file in files:
            if file.filename == '':
                continue

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # PROSES SEGMENTASI
                data = segment_leaf(filepath, k)

                original_rel = f"uploads/{filename}"
                result_rel = clean_static_path(data["result_path"])
                bar_rel = clean_static_path(data["bar_chart"])

                for c in data["clusters"]:
                    c["path"] = clean_static_path(c["path"])

                all_results.append({
                    "original": original_rel,
                    "result": result_rel,
                    "bar_chart": bar_rel,
                    "clusters": data["clusters"],
                    "numeric_data": data["numeric_data"],
                    "dominant_color": data["dominant_color"],
                    "dominant_status": data["dominant_status"],
                    "rf_leaf_prediction": data.get("rf_leaf_prediction"),
                    "conclusion": data.get("conclusion"),
                    "short_conclusion": data.get("short_conclusion")
                })

        # COMPARATIVE SUMMARY (jika >1 gambar)
        comparison = None
        if len(all_results) > 1:
            counts = {}
            probs = {}
            total = len(all_results)
            for r in all_results:
                pred = r.get("rf_leaf_prediction") or {}
                label = (pred.get("label") or "unknown").lower()
                prob = float(pred.get("prob") or 0.0)
                counts[label] = counts.get(label, 0) + 1
                probs.setdefault(label, []).append(prob)

            # compute percentages and avg probs
            summary = []
            for label, cnt in counts.items():
                avg_prob = round((sum(probs.get(label, [])) / len(probs.get(label, []))) if len(probs.get(label, []))>0 else 0.0, 2)
                perc = round((cnt/total)*100, 2)
                summary.append({"label": label.capitalize(), "count": cnt, "percent": perc, "avg_prob": avg_prob})

            # find majority if any
            majority = max(counts.items(), key=lambda x: x[1])
            majority_text = f"Mayoritas daun: {majority[0].capitalize()} ({majority[1]} dari {total})"

            comparison = {"total": total, "breakdown": summary, "majority": majority_text}

        return render_template(
            'result.html',
            results=all_results,
            k=k,
            comparison=comparison
        )

    return render_template('index.html', k=10)

if __name__ == '__main__':
    app.run(debug=True)
