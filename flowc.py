from graphviz import Digraph

def generate_leaf_system_flowchart(output_name="flowchart_sistem_daun"):
    dot = Digraph(
        name="Leaf Segmentation & Classification System",
        format="png"
    )
    dot.attr(rankdir="TB", size="8,12")

    # =========================
    # NODE DEFINITIONS
    # =========================

    dot.node("A", "Start", shape="oval")

    dot.node("B", "User membuka index.html\nUpload gambar & tentukan K",
             shape="parallelogram")

    dot.node("C", "POST request ke app.py\nSimpan gambar ke uploads",
             shape="rectangle")

    dot.node("D", "Model Random Forest tersedia?",
             shape="diamond")

    dot.node("E", "Train Model Random Forest",
             shape="rectangle")

    dot.node("F", "Load Model Random Forest",
             shape="rectangle")

    dot.node("G", "Loop setiap gambar",
             shape="rectangle")

    dot.node("H", "Remove background (rembg)\nHasil RGBA",
             shape="rectangle")

    dot.node("I", "Refine alpha mask\n(Morfologi + Largest CC)",
             shape="rectangle")

    dot.node("J", "Ekstraksi piksel daun",
             shape="rectangle")

    dot.node("K", "K-Means Clustering (K)",
             shape="rectangle")

    dot.node("L", "Hitung centroid RGB\nRule-based labeling",
             shape="rectangle")

    dot.node("M", "Hitung persentase klaster\nGenerate gambar & bar chart",
             shape="rectangle")

    dot.node("N", "Predict cluster (RF)\nAgregasi probabilitas",
             shape="rectangle")

    dot.node("O", "Kesimpulan per gambar",
             shape="rectangle")

    dot.node("P", "Lebih dari satu gambar?",
             shape="diamond")

    dot.node("Q", "Agregasi antar gambar\n(Rata-rata & perbandingan)",
             shape="rectangle")

    dot.node("R", "Render result.html\nTampilkan semua hasil",
             shape="parallelogram")

    dot.node("S", "End", shape="oval")

    # =========================
    # EDGE DEFINITIONS
    # =========================

    dot.edge("A", "B")
    dot.edge("B", "C")
    dot.edge("C", "D")

    dot.edge("D", "E", label="Tidak")
    dot.edge("E", "F")

    dot.edge("D", "F", label="Ya")

    dot.edge("F", "G")
    dot.edge("G", "H")
    dot.edge("H", "I")
    dot.edge("I", "J")
    dot.edge("J", "K")
    dot.edge("K", "L")
    dot.edge("L", "M")
    dot.edge("M", "N")
    dot.edge("N", "O")

    dot.edge("O", "P")
    dot.edge("P", "Q", label="Ya")
    dot.edge("P", "R", label="Tidak")

    dot.edge("Q", "R")
    dot.edge("R", "S")

    # =========================
    # EXPORT
    # =========================
    dot.render(output_name, cleanup=True)
    print(f"Flowchart berhasil dibuat: {output_name}.png")


if __name__ == "__main__":
    generate_leaf_system_flowchart()
