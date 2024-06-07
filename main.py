import cv2
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

cut_top = 50
normalized_shape = (24, 24)


def img_to_bin(image):
    # convert image to black and white
    bin_img = cv2.cvtColor(image[cut_top:-cut_top], cv2.COLOR_BGR2GRAY)
    _, bin_image = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY)
    bin_image = 255 - bin_image

    return bin_image


def bin_image_to_components(bin_image):
    _, _, stats, _ = cv2.connectedComponentsWithStats(bin_image)
    rows = {}
    for x, y, w, h, area in stats[1:]:
        grid_y = y // 24
        if grid_y not in rows:
            rows[grid_y] = []
        rows[grid_y].append((x, y, w, h, area))

    return np.array(list(rows.values()))


def pca(X, n_components):
    # Standardize data
    X = X.astype(np.float32)
    X_mean = tf.reduce_mean(X, axis=0)
    X_centered = X - X_mean

    # Calculate covariance matrix
    cov_matrix = tf.matmul(X_centered, X_centered, transpose_a=True) / tf.cast(
        tf.shape(X_centered)[0] - 1, tf.float32
    )

    # Eigen decomposition
    eigenvalues, eigenvectors = tf.linalg.eigh(cov_matrix)

    # Select top 2 principal components
    top_k_eigenvectors = eigenvectors[:, -n_components:]
    X_pca = tf.matmul(X_centered, top_k_eigenvectors)

    return X_pca.numpy()


def find_sequences(grid, sequence) -> list[list[tuple[int, int]]]:
    rows, cols = grid.shape
    seq_len = len(sequence)

    def check_direction(r, c, dr, dc):
        positions = []
        # print(f"checking dir {r = } {c = } {dr = } {dc = }")
        for i in range(seq_len):
            nr = r + dr * i
            nc = c + dc * i
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                return None
            if grid[nr, nc] != sequence[i]:
                return None
            positions.append((nr, nc))
        return positions

    sequences = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == sequence[0]:
                # Check all 8 directions
                directions = [
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ]
                for dr, dc in directions:
                    found_sequence = check_direction(r, c, dr, dc)
                    if found_sequence:
                        sequences.append(found_sequence)
    return sequences


def apply_red_tint(image, top_left, bottom_right, alpha=0.2):
    # Define the rectangle coordinates
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Create a red overlay
    overlay = image.copy()
    cv2.rectangle(
        overlay, (x1, y1), (x2, y2), (0, 0, 255), -1
    )  # BGR format (Blue, Green, Red)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image


if __name__ == "__main__":

    image_path = "kayak.png"
    print("Loading image...")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    print("Converting image to grayscale")
    bin_image = img_to_bin(image)

    print("Finding components...")
    component_grid = bin_image_to_components(bin_image)
    print(f"{component_grid.shape = }")
    grid_shape = component_grid.shape[:-1]

    print("Reshaping grid to be flat...")
    flat_grid = component_grid.reshape(-1, component_grid.shape[-1])
    print(f"{flat_grid.shape = }")

    def stats_to_img(stats):
        x, y, w, h, _ = stats
        img = bin_image[y : y + h, x : x + w]
        img = cv2.resize(img, normalized_shape)
        return img.flatten()

    print("Creating image component grid...")
    component_img_grid = np.apply_along_axis(stats_to_img, 1, flat_grid)
    print(f"{component_img_grid.shape = }")

    print("Applying PCA...")
    X_pca = pca(component_img_grid, n_components=2)
    print(f"{X_pca.shape = }")
    kmeans = KMeans(n_clusters=3)
    print("Identifying clusters...")
    clusters = kmeans.fit_predict(X_pca)

    classes = {
        clusters[0]: "A",
        clusters[2]: "Y",
        clusters[3]: "K",
    }

    print("Creating character grid...")
    char_grid = clusters.reshape(grid_shape)
    char_grid = np.vectorize(classes.get)(char_grid)
    print(f"{char_grid.shape = }")

    sequence = "KAYAK"
    print(f"Finding {sequence!r} sequences...")
    sequences = find_sequences(char_grid, sequence=sequence)
    if sequences:
        for positions in sequences:
            print(f"Sequence found at positions: {positions}")
    else:
        print("Sequence not found.")

    res_img = image.copy()
    for found_sequence in sequences:
        for r, c in found_sequence:
            x, y, w, h, area = component_grid[r, c]
            y += cut_top
            apply_red_tint(res_img, (x, y), (x + w, y + h))

    cv2.imwrite("solution.png", res_img)
