# K‑Means Image Segmentation & Compression (from scratch)
Adaptive Color Quantization for Image Compression

> A clear, representation of **k‑means clustering** for image segmentation and color compression. Pixels are clustered in RGB space; each pixel is replaced by its cluster’s color to reconstruct a compressed image. Includes two init strategies (random, k‑means++‑style spaced), multiple **k** values, and quantitative evaluation with **MSE**.

---

## 📸 Quick look — inputs & outputs (placeholders)
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/5a3ac8fa-85d0-4339-8e5f-ba81ec9f65cb" width="550" height="850" alt="Original image – cat">
      <br>
      <p align="center"><b>Fig 1.</b></p>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/6253f4ad-f406-47f0-873c-68951fdee499" width="525" height="770" alt="Original image – car">
      <br>
      <p align="center"><b>Fig 2.</b></p>
    </td>
  </tr>
</table>

---

## 🧠 Plain-English idea

An image is a grid of pixels; each pixel has an RGB color like a point in 3-D space. **k-means** groups similar colors into **k clusters**. After clustering, every pixel is repainted with its cluster’s mean color. Using only **k** colors:

* **Segments** the image into coherent color regions (sky vs. leaves vs. fur), and
* **Compresses** the palette (fewer distinct colors → smaller representation).

This is handy for posterized aesthetics, previews, and as a preprocessing step for other vision tasks.

---

## ✍️ What the code does (at a glance)

* Loads two images (e.g., `cat.jpg`, `car.jpg`).
* Flattens each image to a matrix `X` of shape `(num_pixels, 3)` with one RGB row per pixel.
* Runs **k-means** for multiple `k` values: `[2, 3, 10, 20, 40]`.
* For each `k`, tries three initializations: **random** (twice) and **spaced** (k-means++-style).
* Reconstructs the compressed image and saves it to `results/…`.
* Computes **MSE** between original and reconstruction and logs it to `results.txt`.

> Code landmarks: `initialize_centers_random`, `initialize_centers_spaced`, `kmeans`, `reconstruct_image`, `compute_mse`.

---

## 🔬 From intuition to math

### Objective (what k-means minimizes)

Given data points (pixel colors) $x \in \mathbb{R}^3$, choose $k$ centroids $\{\mu_i\}_{i=1}^k$ and assignments to minimize the within-cluster sum of squares (**WCSS**):

$$
\mathrm{WCSS} \;=\; \sum_{i=1}^{k} \; \sum_{x\in C_i} \,\lVert x - \mu_i \rVert^2 .
$$

Here $C_i$ is the set of points assigned to centroid $i$, and $\lVert\cdot\rVert$ is the Euclidean norm. Smaller WCSS means tighter clusters.

### The two steps that repeat (Lloyd’s algorithm)

1. **Assign:**

$$
x \;\mapsto\; \arg\min_i \; \lVert x - \mu_i \rVert^2 .
$$

2. **Update:**

$$
\mu_i \;\leftarrow\; \frac{1}{|C_i|}\sum_{x\in C_i} x .
$$

Repeat until centroids barely move (or a max-iteration cap). Each cycle never increases WCSS, so it converges to a local minimum.

### Why initialization matters

k-means can land in different local minima depending on the starting centroids.

* **Random init:** pick $k$ random pixels.
* **Spaced init (k-means++-style):** pick the first pixel randomly; pick the rest with probability proportional to squared distance from the nearest chosen center. This spreads seeds, typically improving quality and speed.

---

## 🧩 Reconstructing and evaluating the image

After clustering, reconstruct by replacing each pixel with its cluster mean color:

$$
\text{recon}(p) \;=\; \mu_{\text{label}(p)} \in \mathbb{R}^3 .
$$

Measure fidelity with **Mean Squared Error (MSE)** between the original $I$ and reconstruction $\hat I$:

$$
\mathrm{MSE} \;=\; \frac{1}{N}\sum_{u=1}^{H}\sum_{v=1}^{W}\sum_{c\in\{R,G,B\}} \big(I_{u,v,c}-\hat I_{u,v,c}\big)^2,\quad N=H\cdot W\cdot 3.
$$

Lower MSE ⇒ closer match (though perception doesn’t always track MSE perfectly).

---

## 🛠️ Implementation details (how it maps to code)

* **Data prep:** `Image.open(...).convert('RGB')` → `np.array(...)` → reshape to `(num_pixels, 3)`.
* **Init strategies:**

  * `initialize_centers_random(X, k)`: uniform sample of `k` pixels.
  * `initialize_centers_spaced(X, k)`: first center random; subsequent centers sampled with probability ∝ distance² to the nearest chosen center.
* **Main loop:** `kmeans(X, k, init_strategy, max_iters=100)` repeats assign/update until convergence.
* **Outputs:** `reconstruct_image(...)` saves PNGs; `compute_mse(...)` appends numbers to `results.txt`.
* **Speed:** vectorized NumPy for distance and assignment.

> Outputs live under `results/image_1/` and `results/image_2/` (one PNG per `(k, init)` plus a `results.txt` summary).

---

## 📈 What to expect as k changes

* **Small k (2–3):** strong “posterization”; big flat regions; higher MSE.
* **Moderate k (10–20):** preserves structure and many textures; MSE drops notably.
* **Large k (40+):** very close to original; diminishing returns vs. complexity.

Initialization influences both quality and runtime. Spaced seeding often converges faster and may reduce MSE, but the best choice depends on the image.

---

## 🤔 Common questions (brief)

* **Why RGB space?** Simple and effective for color-driven segmentation; Lab/HSV can be better perceptually.
* **Does k-means find the global best?** No, it finds a local optimum—hence multiple restarts.
* **Is MSE ideal?** It’s standard and fast, but SSIM may align better with human perception.

---

## 🧭 Extensions

* Implement canonical **k-means++** seeding (this code already mirrors the idea).
* Try **Lab** space for more perceptual grouping.
* Add spatial features $(x,y)$ to discourage speckle.
* Auto-choose $k$ via the elbow method or information criteria.

---

## 🗂️ Repository pointers

* `SL4_A3 (2).py` — full implementation (init strategies, loop, reconstruction, MSE logging).
* `SL4_A3_Report.pdf` — write-up with theory, experiments, discussion.

---

## License

MIT — see `LICENSE`.

---


