#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import orthogonal_mp
from sklearn.model_selection import train_test_split


def find_hsi_array(mat_dict):
    candidates = []
    for k, v in mat_dict.items():
        if k.startswith('__'):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 3:
            candidates.append((k, v))
    if not candidates:
        raise ValueError('No 3D array found in .mat file')
    # Prefer common names.
    for preferred in ['cube', 'hsi', 'HSI', 'img', 'data', 'rad', 'reflectance']:
        for k, v in candidates:
            if k == preferred:
                return v.astype(np.float64), k
    # Otherwise pick the 3D array with the smallest dimension likely being bands.
    candidates.sort(key=lambda kv: min(kv[1].shape))
    k, v = candidates[0]
    return v.astype(np.float64), k


def ensure_bands_last(arr):
    # Expected shapes usually HxWxB or BxHxW. Convert to HxWxB.
    shape = arr.shape
    if shape[0] <= 64 and shape[1] > 64 and shape[2] > 64:
        arr = np.transpose(arr, (1, 2, 0))
    elif shape[2] <= 64:
        pass
    else:
        # choose smallest axis as spectral dimension
        band_axis = int(np.argmin(shape))
        if band_axis != 2:
            axes = [0, 1, 2]
            axes.pop(band_axis)
            axes.append(band_axis)
            arr = np.transpose(arr, axes)
    return arr


def resample_bands(hsi, target_bands=31):
    h, w, b = hsi.shape
    if b == target_bands:
        return hsi
    x_old = np.linspace(0.0, 1.0, b)
    x_new = np.linspace(0.0, 1.0, target_bands)
    flat = hsi.reshape(-1, b)
    out = np.empty((flat.shape[0], target_bands), dtype=np.float64)
    for i in range(flat.shape[0]):
        out[i] = np.interp(x_new, x_old, flat[i])
    return out.reshape(h, w, target_bands)


def default_response_matrix(num_bands):
    # Smooth approximate RGB sensitivities over normalized wavelength axis.
    x = np.linspace(0.0, 1.0, num_bands)
    def gauss(mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    r = gauss(0.75, 0.13)
    g = gauss(0.50, 0.11)
    b = gauss(0.25, 0.10)
    R = np.vstack([r, g, b])
    R /= np.maximum(R.sum(axis=1, keepdims=True), 1e-12)
    return R


def load_response_matrix(path, num_bands):
    if path is None:
        return default_response_matrix(num_bands)
    arr = np.load(path)
    if arr.shape != (3, num_bands):
        raise ValueError(f'Response matrix must have shape (3, {num_bands}), got {arr.shape}')
    return arr.astype(np.float64)


def sample_pixels_from_cube(hsi, max_pixels, rng):
    flat = hsi.reshape(-1, hsi.shape[-1])
    n = flat.shape[0]
    if n <= max_pixels:
        return flat
    idx = rng.choice(n, size=max_pixels, replace=False)
    return flat[idx]


def normalize_columns(D):
    D = D.copy()
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    D /= norms
    return D


def omp_batch(D_rgb, C, sparsity):
    # D_rgb: (3, K), C: (N, 3), returns W: (K, N)
    X = D_rgb
    Y = C.T
    W = orthogonal_mp(X, Y, n_nonzero_coefs=sparsity)
    if W.ndim == 1:
        W = W[:, None]
    return W


def ksvd(Y, n_atoms=256, sparsity=8, n_iter=8, rng=None):
    # Y shape: (bands, n_samples)
    bands, n_samples = Y.shape
    if rng is None:
        rng = np.random.default_rng(0)
    if n_samples < n_atoms:
        raise ValueError('n_samples must be >= n_atoms')
    atom_idx = rng.choice(n_samples, size=n_atoms, replace=False)
    D = Y[:, atom_idx].copy()
    D = normalize_columns(D)

    for _ in range(n_iter):
        W = orthogonal_mp(D, Y, n_nonzero_coefs=sparsity)
        if W.ndim == 1:
            W = W[:, None]
        for k in range(n_atoms):
            omega = np.flatnonzero(np.abs(W[k]) > 1e-12)
            if omega.size == 0:
                replacement = Y[:, rng.integers(0, n_samples)]
                D[:, k] = replacement / max(np.linalg.norm(replacement), 1e-12)
                continue
            Wk = W[:, omega].copy()
            Wk[k, :] = 0.0
            Ek = Y[:, omega] - D @ Wk
            U, s, Vt = np.linalg.svd(Ek, full_matrices=False)
            D[:, k] = U[:, 0]
            W[k, omega] = s[0] * Vt[0]
        D = normalize_columns(D)
    W = orthogonal_mp(D, Y, n_nonzero_coefs=sparsity)
    if W.ndim == 1:
        W = W[:, None]
    return D, W


def rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def rrmse(pred, gt):
    denom = np.sqrt(np.mean(gt ** 2))
    return float(np.sqrt(np.mean((pred - gt) ** 2)) / max(denom, 1e-12))


def sam(pred, gt):
    dot = np.sum(pred * gt, axis=1)
    pn = np.linalg.norm(pred, axis=1)
    gn = np.linalg.norm(gt, axis=1)
    cosang = dot / np.maximum(pn * gn, 1e-12)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.mean(np.arccos(cosang)))


def psnr(pred, gt, data_range=1.0):
    mse = np.mean((pred - gt) ** 2)
    return float(20.0 * np.log10(data_range / math.sqrt(max(mse, 1e-12))))


def load_dataset(folder, target_bands=31):
    files = sorted(Path(folder).glob('*.mat'))
    if not files:
        raise FileNotFoundError(f'No .mat files found in {folder}')
    cubes = []
    names = []
    for f in files:
        mat = loadmat(f)
        arr, key = find_hsi_array(mat)
        arr = ensure_bands_last(arr)
        arr = resample_bands(arr, target_bands=target_bands)
        arr = arr.astype(np.float64)
        if arr.max() > 1.0:
            arr /= arr.max()
        cubes.append(arr)
        names.append((f.name, key, arr.shape))
    return cubes, names


def main():
    p = argparse.ArgumentParser(description='Test Arad & Ben-Shahar sparse recovery method on KAUST .mat cubes.')
    p.add_argument('--dataset_root', required=True)
    p.add_argument('--output_json', default='kaust_sparse_recovery_results.json')
    p.add_argument('--bands', type=int, default=31)
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--sample_pixels_per_train_cube', type=int, default=4000)
    p.add_argument('--sample_pixels_per_test_cube', type=int, default=12000)
    p.add_argument('--n_atoms', type=int, default=256)
    p.add_argument('--sparsity', type=int, default=8)
    p.add_argument('--ksvd_iters', type=int, default=8)
    p.add_argument('--response_matrix_npy', default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    cubes, names = load_dataset(args.dataset_root, target_bands=args.bands)
    idx = np.arange(len(cubes))
    train_idx, test_idx = train_test_split(idx, train_size=args.train_ratio, random_state=args.seed)

    train_pixels = []
    for i in train_idx:
        train_pixels.append(sample_pixels_from_cube(cubes[i], args.sample_pixels_per_train_cube, rng))
    train_pixels = np.vstack(train_pixels)

    Dh, _ = ksvd(train_pixels.T, n_atoms=args.n_atoms, sparsity=args.sparsity, n_iter=args.ksvd_iters, rng=rng)
    R = load_response_matrix(args.response_matrix_npy, args.bands)
    D_rgb = R @ Dh

    per_scene = []
    all_pred = []
    all_gt = []

    for i in test_idx:
        gt = sample_pixels_from_cube(cubes[i], args.sample_pixels_per_test_cube, rng)
        rgb = gt @ R.T
        W = omp_batch(D_rgb, rgb, args.sparsity)
        pred = (Dh @ W).T
        pred = np.clip(pred, 0.0, 1.0)

        scene_metrics = {
            'file': names[i][0],
            'mat_key': names[i][1],
            'shape': list(names[i][2]),
            'rmse': rmse(pred, gt),
            'rrmse': rrmse(pred, gt),
            'sam_rad': sam(pred, gt),
            'psnr': psnr(pred, gt),
            'num_test_pixels': int(gt.shape[0]),
        }
        per_scene.append(scene_metrics)
        all_pred.append(pred)
        all_gt.append(gt)

    all_pred = np.vstack(all_pred)
    all_gt = np.vstack(all_gt)
    summary = {
        'config': vars(args),
        'num_cubes': len(cubes),
        'num_train_cubes': int(len(train_idx)),
        'num_test_cubes': int(len(test_idx)),
        'dictionary_shape': list(Dh.shape),
        'rgb_dictionary_shape': list(D_rgb.shape),
        'metrics_mean': {
            'rmse': rmse(all_pred, all_gt),
            'rrmse': rrmse(all_pred, all_gt),
            'sam_rad': sam(all_pred, all_gt),
            'psnr': psnr(all_pred, all_gt),
        },
        'per_scene': per_scene,
    }

    out = Path(args.output_json)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary['metrics_mean'], indent=2, ensure_ascii=False))
    print(f'Wrote full results to {out}')


if __name__ == '__main__':
    main()
