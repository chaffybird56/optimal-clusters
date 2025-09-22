# K‑Means Image Segmentation & Compression (from scratch)
Adaptive Color Quantization for Image Compression

> A clear, representation of **k‑means clustering** for image segmentation and color compression. Pixels are clustered in RGB space; each pixel is replaced by its cluster’s color to reconstruct a compressed image. Includes two init strategies (random, k‑means++‑style spaced), multiple **k** values, and quantitative evaluation with **MSE**.

---

## 📸 Quick look — inputs & outputs (placeholders)

<div align="center">
  <img src="https://github.com/user-attachments/assets/5a3ac8fa-85d0-4339-8e5f-ba81ec9f65cb" width="753" height="1180" alt="Original image – cat"/>
  <br/>
  <sub><b>Fig 1.</sub>

<div align="center">
  <img src="https://github.com/user-attachments/assets/6253f4ad-f406-47f0-873c-68951fdee499" width="753" height="1180" alt="Original image – car"/>
  <br/>
  <sub><b>Fig 2.</sub>
</div>


---

🧠 Plain‑English idea

An image is a grid of pixels; each pixel has an RGB color like a point in 3‑D space. k‑means groups similar colors together into k clusters. After clustering, every pixel is painted with its cluster’s average color. The result uses only k colors, which:

Segments the image into coherent color regions (sky vs. leaves vs. fur), and

Compresses the color palette (fewer distinct colors → smaller representation).

This is useful when a simpler, flatter color representation is enough (posters, icons, previews) or as a preprocessing step for other vision tasks.

✍️ What the code does (at a glance)

Loads two images (e.g., cat.jpg, car.jpg).

Flattens each image to a matrix X of shape (num_pixels, 3) with one RGB row per pixel.

Runs k‑means for multiple k values: [2, 3, 10, 20, 40].

For each k, runs three initializations: random (twice) and spaced (k‑means++‑style).

Reconstructs the compressed image and saves it to results/….

Computes MSE between original and reconstruction and logs it in results.txt.

Key functions (see code): initialize_centers_random, initialize_centers_spaced (k‑means++‑inspired), kmeans, reconstruct_image, compute_mse.

🔬 From intuition to math

Objective (what k‑means is trying to minimize)

Given data points (here, pixel colors) , choose  centroids  and assignments so that points are close to their cluster’s centroid. The within‑cluster sum of squares (WCSS) is



where  is the set of points assigned to cluster , and  is the Euclidean norm. Smaller WCSS means tighter, more coherent clusters.

The two steps that repeat (Lloyd’s algorithm)

Assign each pixel to its nearest centroid:



Update each centroid to the mean of its assigned points:



Repeat until centroids stop moving (or a max number of iterations is reached). Each repeat never increases WCSS, so the process converges to a local minimum.

Why initialization matters

k‑means can settle in different local minima depending on the starting centroids.

Random init: pick  random pixels.

Spaced init (k‑means++‑style): choose the first pixel randomly; then probabilistically pick new centroids far from those already chosen (proportional to squared distance). This spreads initial centers and often speeds convergence / improves quality.

🧩 Reconstructing and evaluating the image

After clustering, rebuild the image by replacing each pixel by its cluster mean color:



To quantify fidelity, compute Mean Squared Error (MSE) between the original image  and the reconstruction :



where . Lower MSE  better reconstruction (but human perception doesn’t always align perfectly with MSE).

🛠️ Implementation details (mapping to the code)

Data prep: Image.open(...).convert('RGB') → np.array(...) → reshape to (num_pixels, 3).

Init strategies:

initialize_centers_random(X, k): uniform random sample of k pixels.

initialize_centers_spaced(X, k): first center random; subsequent centers sampled with probability  where  is distance to the nearest chosen center (k‑means++ logic).

Main loop: kmeans(X, k, init_strategy, max_iters=100) repeatedly assigns labels and recomputes centroids until convergence.

Outputs: reconstruct_image(labels, centers, image_shape) → save PNG; compute_mse(...) → log to results.txt along with iteration count.

Batching: the code uses vectorized NumPy operations for distances and assignment for speed.

Tip: results are written under results/image_1/ and results/image_2/, one PNG per (k, init) plus a results.txt summary.

📈 What to expect as k changes

Small k (e.g., 2–3): strong “posterization” (large, flat color regions). Compression is high; detail is lost. MSE is usually higher.

Moderate k (e.g., 10–20): preserves large structures and many textures; good trade‑off. MSE drops notably.

Large k (e.g., 40+): close to the original; diminishing returns in MSE vs. added complexity.

Initialization impacts quality/time mildly. Spaced init often converges in fewer iterations and can reduce MSE, but the best choice depends on image content.

📌 Results gallery (placeholders)

Use a few representative reconstructions to keep the README readable. Suggested layout:

Image 1 (cat)

Image 2 (car)

Swap in your uploaded images from results/.... Keep widths modest (≤500) so the page stays readable.

🤔 Common questions (brief)

Why RGB space? It’s simple and works well for color‑driven segmentation. Alternatives like Lab/HSV can sometimes cluster perceptually better.

Will k‑means always find the best clustering? No—only a local optimum. That’s why initialization (and multiple restarts) can help.

Is MSE the best metric? It’s convenient and standard, but perceptual metrics (SSIM) may correlate better with human judgment.

🧭 Extensions

k‑means++ exactly: use the canonical seeding (this code already mirrors the idea).

Color spaces: run in Lab for perceptual uniformity.

Spatial regularization: add pixel position () to the feature vector to discourage speckle.

Auto‑choose k: elbow method or information criteria.

🗂️ Repository pointers

SL4_A3 (2).py — full implementation (init strategies, k‑means loop, reconstruction, MSE logging).

SL4_A3_Report.pdf — write‑up with theory, experiments, and discussion.

License

MIT — see LICENSE.



