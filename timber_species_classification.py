import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# ─── Dataset Path ──────────────────────────────────────────────────────────────
dataset_path = r"C:\Users\Dell\Desktop\3rd year\my_lec\3141\project\dataset"
species_names = ["Coconut","London_Plane","Pine","Rainbow_Eucalyptus"]

# ─── Feature Extraction Function ───────────────────────────────────────────────
def extract_features(img):

    # Resize
    img = cv2.resize(img, (256, 256))

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gamma Correction
    gamma = np.uint8(255 * (gray / 255) ** 0.8)

    # Gaussian Blur
    blur = cv2.GaussianBlur(gamma, (5, 5), 0)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
    clahe_img = clahe.apply(blur)

    # Top-Hat Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9, 9))
    tophat = cv2.morphologyEx(clahe_img,cv2.MORPH_TOPHAT,kernel)

    # Otsu Threshold
    _, binary = cv2.threshold(tophat,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Skeletonization
    skeleton = np.zeros(binary.shape, np.uint8)
    temp = binary.copy()
    k_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3))

    while True:
        eroded = cv2.erode(temp, k_cross)
        opened = cv2.dilate(eroded, k_cross)
        skeleton = cv2.bitwise_or(skeleton,cv2.subtract(temp, opened))
        temp = eroded.copy()

        if cv2.countNonZero(temp) == 0:
            break

    # Fourier Transform
    fshift = np.fft.fftshift(np.fft.fft2(tophat))
    magnitude = np.log(np.abs(fshift) + 1)

    # Features
    ridge_density = np.count_nonzero(skeleton) / skeleton.size
    freq_mean = np.mean(np.abs(fshift))
    texture_mean = np.mean(clahe_img)
    texture_std = np.std(clahe_img)

    features = (ridge_density,freq_mean,texture_mean,texture_std)

    stages = {
        "gray": gray,
        "gamma": gamma,
        "blur": blur,
        "clahe": clahe_img,
        "tophat": tophat,
        "binary": binary,
        "skeleton": skeleton,
        "fourier": magnitude
    }

    return features, stages

# ─── Load Dataset ──────────────────────────────────────────────────────────────
features_all = []
labels_all = []

for species in species_names:
    folder = os.path.join(dataset_path, species)
    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        ):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is None:
                continue
            feats, _ = extract_features(img)
            features_all.append(feats)
            labels_all.append(species)

if len(features_all) == 0:
    print("Dataset not found or empty.")
    exit()
print("Dataset loaded:", len(features_all), "images")

# ─── Build Thresholds ──────────────────────────────────────────────────────────
thresholds = {}
for species in species_names:
    vals = [f for f, l in zip(features_all, labels_all) if l == species]
    if len(vals) == 0:
        continue
    arr = np.array(vals)
    thresholds[species] = {
        "ridge":   (arr[:, 0].min(), arr[:, 0].max()),
        "freq":    (arr[:, 1].min(), arr[:, 1].max()),
        "mean":    (arr[:, 2].min(), arr[:, 2].max()),
        "std":     (arr[:, 3].min(), arr[:, 3].max()),
    }

# ─── Classification ────────────────────────────────────────────────────────────
def classify(feats, thresholds):
    rd, fr, tm, ts = feats
    scores = {}
    for species, rule in thresholds.items():
        # Count how many features fall within range → higher = better match
        score = sum([
            rule["ridge"][0] <= rd <= rule["ridge"][1],
            rule["freq"][0] <= fr <= rule["freq"][1],
            rule["mean"][0] <= tm <= rule["mean"][1],
            rule["std"][0] <= ts <= rule["std"][1]])
        scores[species] = score
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Unknown"
def browse_image():

    file_path = filedialog.askopenfilename(
        title="Select Bark Image",
        filetypes=[("Image Files","*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])

    if not file_path:
        return

    img_new = cv2.imread(file_path)

    if img_new is None:
        messagebox.showerror("Error", "Image not found.")
        return

    feats_new, stages = extract_features(img_new)
    result = classify(feats_new, thresholds)

    result_label.config(text=f"Predicted Species : {result}")

    original_rgb = cv2.cvtColor(cv2.resize(img_new,(256,256)),cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,10))

    plt.subplot(3,3,1)
    plt.imshow(original_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(3,3,2)
    plt.imshow(stages["gray"], cmap="gray")
    plt.title("Gray")
    plt.axis("off")

    plt.subplot(3,3,3)
    plt.imshow(stages["gamma"], cmap="gray")
    plt.title("Gamma")
    plt.axis("off")

    plt.subplot(3,3,4)
    plt.imshow(stages["blur"], cmap="gray")
    plt.title("Blur")
    plt.axis("off")

    plt.subplot(3,3,5)
    plt.imshow(stages["clahe"], cmap="gray")
    plt.title("CLAHE")
    plt.axis("off")

    plt.subplot(3,3,6)
    plt.imshow(stages["tophat"], cmap="gray")
    plt.title("Top-Hat")
    plt.axis("off")

    plt.subplot(3,3,7)
    plt.imshow(stages["binary"], cmap="gray")
    plt.title("Otsu")
    plt.axis("off")

    plt.subplot(3,3,8)
    plt.imshow(stages["skeleton"], cmap="gray")
    plt.title("Skeleton")
    plt.axis("off")

    plt.show()

root = tk.Tk()
root.title("Timber Species Classification System")
root.geometry("500x300")
title_label = tk.Label(root,text="Tree Bark Species Identification",font=("Arial",16,"bold"))
title_label.pack(pady=20)
browse_btn = tk.Button(root,text="Browse Bark Image",font=("Arial",12),command=browse_image)
browse_btn.pack(pady=20)
result_label = tk.Label(root,text="Prediction will appear here",font=("Arial",14))
result_label.pack(pady=20)
root.mainloop()
