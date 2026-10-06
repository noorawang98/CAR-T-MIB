import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import re
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import sys
input_dir = Path(sys.argv[1])
output_dir = input_dir / 'spatial_h5ad_results'
output_dir.mkdir(parents=True, exist_ok=True)
file_pattern = '*ROI*.csv'
expr_suffix = ' Cell Intensity Average'
cofactor = 5.0
exclude_patterns = ['^DAPI\\b', '^DAPI C\\d+\\b', 'autofluorescence', 'isotype', 'blank', 'background']
qc_params = {'cell_size_low': 0.005, 'cell_size_high': 0.995, 'nucleus_size_low': 0.005, 'nucleus_size_high': 0.995, 'dapi_low': 0.005, 'dapi_high': 0.995, 'marker_detected_low': 0.0, 'marker_detected_high': 1.0, 'marker_std_min': 0}
leiden_resolution = 0.5
n_neighbors = 15
max_pcs = 20

def read_csv_auto(infile):
    infile = Path(infile)
    if infile.suffix.lower() in ['.tsv', '.txt']:
        return pd.read_csv(infile, sep='\t')
    elif infile.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(infile)
    else:
        return pd.read_csv(infile)

def is_excluded_marker(col):
    marker = col.replace(expr_suffix, '')
    return any((re.search(p, marker, flags=re.IGNORECASE) for p in exclude_patterns))

def between_quantile(s, low=0.01, high=0.99):
    s = pd.to_numeric(s, errors='coerce')
    return s.between(s.quantile(low), s.quantile(high))

def make_safe_name(path):
    """
    用文件名作为 ROI 名称。
    例如 sample_ROI_001.csv -> sample_ROI_001
    """
    return Path(path).stem.replace(' ', '_')

def find_spatial_columns(df):
    """
    自动识别空间坐标列。
    优先使用你的 MACSima 表格中常见的 Cell Center X / Cell Center Y。
    """
    xy_candidates = [('Cell Center X', 'Cell Center Y'), ('Cell Center X', 'Cell Center Y Inverted'), ('Center X', 'Center Y'), ('Cell X Position', 'Cell Y Position'), ('X', 'Y'), ('x', 'y')]
    for x_col, y_col in xy_candidates:
        if {x_col, y_col}.issubset(df.columns):
            return (x_col, y_col)
    return (None, None)

def csv_to_spatial_h5ad(infile, roi_name):
    df = read_csv_auto(infile)
    print('\n' + '=' * 80)
    print(f'Processing ROI file: {infile}')
    print(f'ROI name: {roi_name}')
    print('Input shape:', df.shape)
    expr_cols = [c for c in df.columns if c.endswith(expr_suffix)]
    expr_cols = [c for c in expr_cols if not is_excluded_marker(c)]
    if len(expr_cols) == 0:
        raise ValueError(f"No marker expression columns ending with '{expr_suffix}' found.")
    markers = [c.replace(expr_suffix, '') for c in expr_cols]
    print(f'Using {len(expr_cols)} marker columns')
    print('First markers:', markers[:10])
    X_raw = df[expr_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    X = np.arcsinh(X_raw / cofactor).astype(np.float32)
    obs_cols = [c for c in df.columns if c not in expr_cols]
    obs = df[obs_cols].copy()
    if 'Cell Id' in obs.columns:
        obs.index = roi_name + '_cell_' + obs['Cell Id'].astype(str)
    else:
        obs.index = [f'{roi_name}_cell_{i}' for i in range(df.shape[0])]
    obs.index.name = 'cell_id'
    obs['roi'] = roi_name
    obs['sample'] = roi_name
    obs['source_file'] = str(infile)
    if 'ROI index' not in obs.columns:
        obs['ROI index'] = roi_name
    for c in obs.columns:
        obs[c] = pd.to_numeric(obs[c], errors='ignore')
    var = pd.DataFrame(index=markers)
    var['source_column'] = expr_cols
    var['compartment'] = 'cell'
    var['feature_type'] = 'protein'
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers['raw'] = X_raw
    x_col, y_col = find_spatial_columns(df)
    if x_col is None or y_col is None:
        raise ValueError("No spatial coordinate columns found. Expected columns like 'Cell Center X' and 'Cell Center Y'.")
    coords = df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
    adata.obsm['spatial'] = coords
    adata.obs['x'] = coords[:, 0]
    adata.obs['y'] = coords[:, 1]
    adata.uns['spatial'] = {roi_name: {'coordinate_columns': {'x': x_col, 'y': y_col}}}
    print(f"Added spatial coordinates to adata.obsm['spatial']")
    print(f'Spatial columns: {x_col}, {y_col}')
    if {'Cell Center X', 'Cell Center Y Inverted'}.issubset(df.columns):
        coords_inv = df[['Cell Center X', 'Cell Center Y Inverted']].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
        adata.obsm['spatial_y_inverted'] = coords_inv
        adata.obs['y_inverted'] = coords_inv[:, 1]
    adata.uns['macsima'] = {'roi': roi_name, 'source_file': str(infile), 'expr_suffix': expr_suffix, 'cofactor': cofactor, 'excluded_patterns': exclude_patterns}
    return adata

def qc_filter_adata(adata):
    obs = adata.obs
    qc_mask = np.ones(adata.n_obs, dtype=bool)
    if 'Quality Cell In-Focus' in obs.columns:
        qc_mask &= pd.to_numeric(obs['Quality Cell In-Focus'], errors='coerce').fillna(0).values > 0
    if 'Quality Nuclear Segmentation' in obs.columns:
        qc_mask &= pd.to_numeric(obs['Quality Nuclear Segmentation'], errors='coerce').fillna(0).values > 0
    if 'Cell Size' in obs.columns:
        qc_mask &= between_quantile(obs['Cell Size'], qc_params['cell_size_low'], qc_params['cell_size_high']).fillna(False).values
    if 'Nucleus Size' in obs.columns:
        qc_mask &= between_quantile(obs['Nucleus Size'], qc_params['nucleus_size_low'], qc_params['nucleus_size_high']).fillna(False).values
    if 'DAPI Nucleus Intensity Average' in obs.columns:
        qc_mask &= between_quantile(obs['DAPI Nucleus Intensity Average'], qc_params['dapi_low'], qc_params['dapi_high']).fillna(False).values
    adata.obs['pass_qc'] = qc_mask
    print('Before QC:', adata.n_obs)
    print('After QC:', int(qc_mask.sum()))
    if qc_mask.sum() == 0:
        raise ValueError('No cells passed QC.')
    adata_qc = adata[adata.obs['pass_qc'].values].copy()
    X = np.asarray(adata_qc.X)
    marker_std = X.std(axis=0)
    marker_detected = (X > 0).mean(axis=0)
    keep_marker = np.ones(adata_qc.n_vars, dtype=bool)
    adata_qc.var = adata_qc.var.copy()
    adata_qc.var['marker_std'] = marker_std
    adata_qc.var['marker_detected_fraction'] = marker_detected
    adata_qc.var['keep_marker'] = keep_marker
    if keep_marker.sum() == 0:
        raise ValueError('No markers passed filtering.')
    adata_qc = adata_qc[:, keep_marker].copy()
    print('After marker filtering:', adata_qc.shape)
    return adata_qc

def run_downstream_analysis(adata, resolution=0.5, roi_name=None):
    """
    Run PCA / neighbors / UMAP / Leiden.
    If downstream analysis fails, return None so this ROI can be skipped.
    """
    try:
        adata = adata.copy()
        n_cells = adata.n_obs
        n_markers = adata.n_vars
        prefix = f'[{roi_name}] ' if roi_name is not None else ''
        print(f'{prefix}Downstream input shape: cells={n_cells}, markers={n_markers}')
        if n_cells < 3:
            print(f'{prefix}Too few cells. Skip PCA/UMAP/Leiden.')
            return None
        if n_markers < 3:
            print(f'{prefix}Too few markers. Skip PCA/UMAP/Leiden.')
            return None
        X = np.asarray(adata.X)
        valid_cell_mask = np.isfinite(X).all(axis=1)
        valid_marker_mask = np.isfinite(X).all(axis=0)
        if valid_cell_mask.sum() < n_cells:
            print(f'{prefix}Remove invalid cells: {n_cells - valid_cell_mask.sum()}')
        if valid_marker_mask.sum() < n_markers:
            print(f'{prefix}Remove invalid markers: {n_markers - valid_marker_mask.sum()}')
        adata = adata[valid_cell_mask, valid_marker_mask].copy()
        n_cells = adata.n_obs
        n_markers = adata.n_vars
        if n_cells < 3:
            print(f'{prefix}Too few valid cells after removing NaN/Inf. Skip.')
            return None
        if n_markers < 3:
            print(f'{prefix}Too few valid markers after removing NaN/Inf. Skip.')
            return None
        X = np.asarray(adata.X)
        marker_std = X.std(axis=0)
        keep_marker = marker_std > 1e-08
        if keep_marker.sum() < n_markers:
            print(f'{prefix}Remove zero-variance markers: {n_markers - keep_marker.sum()}')
        adata = adata[:, keep_marker].copy()
        n_cells = adata.n_obs
        n_markers = adata.n_vars
        if n_markers < 3:
            print(f'{prefix}Too few variable markers after filtering. Skip.')
            return None
        sc.pp.scale(adata, max_value=10)
        X = np.asarray(adata.X)
        valid_cell_mask = np.isfinite(X).all(axis=1)
        valid_marker_mask = np.isfinite(X).all(axis=0)
        adata = adata[valid_cell_mask, valid_marker_mask].copy()
        n_cells = adata.n_obs
        n_markers = adata.n_vars
        if n_cells < 3 or n_markers < 3:
            print(f'{prefix}Too few cells/markers after scaling. Skip.')
            return None
        n_pcs = min(max_pcs, n_markers - 1, n_cells - 1)
        if n_pcs < 2:
            print(f'{prefix}n_pcs={n_pcs} < 2. Skip PCA/UMAP/Leiden.')
            return None
        print(f'{prefix}Using n_pcs={n_pcs}')
        sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack')
        n_neighbors_use = min(n_neighbors, n_cells - 1)
        if n_neighbors_use < 2:
            print(f'{prefix}n_neighbors={n_neighbors_use} < 2. Skip UMAP/Leiden.')
            return None
        print(f'{prefix}Using n_neighbors={n_neighbors_use}')
        sc.pp.neighbors(adata, n_neighbors=n_neighbors_use, n_pcs=n_pcs)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution, key_added=f'leiden_{resolution}')
        adata.uns['downstream_status'] = 'success'
        adata.uns['downstream_params'] = {'n_pcs': n_pcs, 'n_neighbors': n_neighbors_use, 'resolution': resolution}
        return adata
    except Exception as e:
        prefix = f'[{roi_name}] ' if roi_name is not None else ''
        print(f'{prefix}Downstream analysis failed. Skip this ROI.')
        print(f'{prefix}Error: {e}')
        return None
roi_files = sorted(input_dir.glob(file_pattern))
print(f'Found {len(roi_files)} ROI csv files in {input_dir}')
if len(roi_files) == 0:
    raise FileNotFoundError(f'No files found with pattern: {file_pattern}')
failed = []
filtered_adatas = []
for roi_file in roi_files:
    roi_name = make_safe_name(roi_file)
    roi_out_dir = output_dir / roi_name
    roi_out_dir.mkdir(parents=True, exist_ok=True)
    raw_h5ad = roi_out_dir / f'{roi_name}_spatial_raw.h5ad'
    filtered_h5ad = roi_out_dir / f'{roi_name}_spatial_filtered.h5ad'
    processed_h5ad = roi_out_dir / f'{roi_name}_spatial_processed.h5ad'
    try:
        adata = csv_to_spatial_h5ad(roi_file, roi_name)
        adata.write_h5ad(raw_h5ad, compression='gzip')
        adata_qc = qc_filter_adata(adata)
        adata_qc.write_h5ad(filtered_h5ad, compression='gzip')
        adata_processed = run_downstream_analysis(adata_qc, resolution=leiden_resolution, roi_name=roi_name)
        if adata_processed is None:
            print(f'Skip ROI from merged analysis: {roi_name}')
            failed.append({'roi': roi_name, 'file': str(roi_file), 'error': 'Downstream PCA/UMAP/Leiden failed or skipped'})
            continue
        adata_processed.write_h5ad(processed_h5ad, compression='gzip')
        filtered_adatas.append(adata_processed)
        print(f'Finished ROI: {roi_name}')
        print(f'Saved raw: {raw_h5ad}')
        print(f'Saved filtered: {filtered_h5ad}')
        print(f'Saved processed: {processed_h5ad}')
    except Exception as e:
        print(f'Failed ROI: {roi_name}')
        print('Error:', e)
        failed.append({'roi': roi_name, 'file': str(roi_file), 'error': str(e)})
if len(filtered_adatas) > 0:
    print('\n' + '=' * 80)
    print('Merging all filtered ROI AnnData objects...')
    filtered_adatas_valid = [a for a in filtered_adatas if a is not None and a.n_obs > 0 and (a.n_vars > 0)]
    if len(filtered_adatas_valid) == 0:
        print('No valid ROI AnnData objects to merge.')
    else:
        roi_keys = [a.obs['roi'].iloc[0] if 'roi' in a.obs.columns else f'ROI_{i}' for i, a in enumerate(filtered_adatas_valid)]
        adata_merged = ad.concat(filtered_adatas_valid, join='inner', label='roi_batch', keys=roi_keys, index_unique=None)
        print('Merged raw shape:', adata_merged.shape)
        print('Merged obsm keys:', list(adata_merged.obsm.keys()))
        if 'spatial' in adata_merged.obsm:
            print('Merged adata has spatial coordinates: True')
            print('Merged spatial shape:', adata_merged.obsm['spatial'].shape)
        else:
            print('Warning: merged adata has no spatial coordinates.')
        X = np.asarray(adata_merged.X)
        marker_std = X.std(axis=0)
        marker_detected = (X > 0).mean(axis=0)
        keep_marker = np.isfinite(marker_std) & (marker_std > 0)
        adata_merged.var = adata_merged.var.copy()
        adata_merged.var['marker_std'] = marker_std
        adata_merged.var['marker_detected_fraction'] = marker_detected
        adata_merged.var['keep_marker'] = keep_marker
        adata_merged = adata_merged[:, keep_marker].copy()
        print('Merged shape after marker filtering:', adata_merged.shape)
        merged_filtered_h5ad = output_dir / 'all_ROI_spatial_filtered_merged.h5ad'
        merged_processed_h5ad = output_dir / 'all_ROI_spatial_processed_merged.h5ad'
        adata_merged.write_h5ad(merged_filtered_h5ad, compression='gzip')
        print(f'Saved merged filtered h5ad: {merged_filtered_h5ad}')
        print('Running downstream analysis on merged object...')
        adata_merged_processed = run_downstream_analysis(adata_merged, resolution=leiden_resolution, roi_name='merged_all_ROI')
        if adata_merged_processed is None:
            print('Merged downstream analysis failed or skipped.')
            print(f'Only merged filtered h5ad was saved: {merged_filtered_h5ad}')
        else:
            adata_merged_processed.write_h5ad(merged_processed_h5ad, compression='gzip')
            print(f'Saved merged processed h5ad: {merged_processed_h5ad}')
else:
    print('No successfully processed ROI files to merge.')
'\n# ============================================================\n# 6. 合并所有 ROI，并统一做下游分析\n# ============================================================\n\nif len(filtered_adatas) > 0:\n    print("\n" + "=" * 80)\n    print("Merging all filtered ROI AnnData objects...")\n\n    adata_merged = ad.concat(\n        filtered_adatas,\n        join="inner",          # 只保留所有 ROI 共有 marker\n        label="roi_batch",\n        keys=[a.obs["roi"].iloc[0] for a in filtered_adatas],\n        index_unique=None,\n    )\n\n    # 确保 spatial 仍然存在\n    if "spatial" in filtered_adatas[0].obsm:\n        print("Merged adata has spatial coordinates:", "spatial" in adata_merged.obsm)\n\n    X = np.asarray(adata_merged.X)\n\n    marker_std = X.std(axis=0)\n    marker_detected = (X > 0).mean(axis=0)\n    \n    keep_marker = (\n        np.isfinite(marker_std) &\n        (marker_std > 0)\n    )\n    \n    adata_merged.var["marker_std"] = marker_std\n    adata_merged.var["marker_detected_fraction"] = marker_detected\n    adata_merged.var["keep_marker"] = keep_marker\n    \n    adata_merged = adata_merged[:, keep_marker].copy()\n    \n    print("Merged shape after marker filtering:", adata_merged.shape)\n    merged_filtered_h5ad = output_dir / "all_ROI_spatial_filtered_merged.h5ad"\n    merged_processed_h5ad = output_dir / "all_ROI_spatial_processed_merged.h5ad"\n\n    adata_merged.write_h5ad(merged_filtered_h5ad, compression="gzip")\n\n    print("Merged shape:", adata_merged.shape)\n    print("Running downstream analysis on merged object...")\n\n    adata_merged_processed = run_downstream_analysis(\n        adata_merged,\n        resolution=leiden_resolution,\n        roi_name="merged_all_ROI",\n    )\n    if adata_merged_processed is None:\n        print("Merged downstream analysis failed or skipped.")\n        print(f"Only merged filtered h5ad was saved: {merged_filtered_h5ad}")\n    else:\n        adata_merged_processed.write_h5ad(\n            merged_processed_h5ad,\n            compression="gzip",\n        )\n\n\n    adata_merged_processed.write_h5ad(\n        merged_processed_h5ad,\n        compression="gzip",\n    )\n\n    print(f"Saved merged filtered h5ad: {merged_filtered_h5ad}")\n    print(f"Saved merged processed h5ad: {merged_processed_h5ad}")\n\nelse:\n    print("No successfully processed ROI files to merge.")\n\n'
if failed:
    failed_df = pd.DataFrame(failed)
    failed_csv = output_dir / 'failed_ROI_files.csv'
    failed_df.to_csv(failed_csv, index=False)
    print(f'\nFailed {len(failed)} ROI files. See: {failed_csv}')
else:
    print('\nAll ROI files processed successfully.')
