from imports import *


class EncodingModel:
    def __init__(
        self,
        data_dir: str,
        verbose: bool = True,
    ):
        """
        A lightweight data container for movie-based ROI and rating data.
        data_dir: root directory (e.g., './Movie_ROI')
        """
        self.data_dir = str(data_dir).rstrip("/")
        self.verbose = verbose

        # ROI data
        self.roi_all = None           # (num_subjects, n_roi, T)
        self.roi_subject_ids = None   # list[str]

        # Rating data
        self.rating_cube = None       # (num_subjects, N, 2)
        self.rating_subjects = None   # list[str]

        # Atlas / groupings
        self.atlas_df = None  # ['Atlas','Hemisphere','Subnetwork','Region','Parcel index','ROI ID','R','A','S']
        self.roi_by_network_hemisphere = None         # {Subnetwork: {LH/RH: [roi_ids]}}
        self.roi_by_atlas_network_hemi = None         # {Atlas: {Subnetwork: {LH/RH: [roi_ids]}}}

        # Feature containers
        self.feature_dict = {}        # {feature_name: np.ndarray of shape (T, d)}
        self.feature_names = []       # load order (preserves user-specified order)


    # ------------------- ROI LOADER ------------------- #
    def load_roi_cube(
        self,
        folder: str,
        pattern: str = "sub-*_ROI114.csv",
        dtype=np.float32,
        fillna: bool = True,
    ):
        """
        Read ROI CSVs matching a pattern and stack them into self.roi_all.
        Handles both (T,114) and (114,T) CSVs; normalized to (114,T).
        Stores subject IDs in self.roi_subject_ids.
        """
        folder = Path(folder)
        files = sorted(folder.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No ROI files matched: {folder / pattern}")

        arrays, subject_ids = [], []
        for fp in files:
            df = pd.read_csv(fp)
            arr = df.to_numpy(dtype=dtype)

            # Normalize to (114, T)
            if arr.shape[0] != 114 and arr.shape[1] == 114:
                arr = arr.T
            elif arr.shape[0] != 114:
                arr = arr.T

            if fillna:
                arr = np.nan_to_num(arr)

            arrays.append(arr)
            # Extract subject ID from filename (sub-XXXX)
            m = re.search(r"sub-([A-Za-z0-9]+)", fp.name)
            subject_ids.append(m.group(1) if m else fp.stem)

        self.roi_all = np.stack(arrays, axis=0)
        self.roi_subject_ids = subject_ids

        if self.verbose:
            print("[load_roi_cube] roi_all shape:", self.roi_all.shape)
            print("[load_roi_cube] first 3 subjects:", subject_ids[:3])


    # ------------------- RATING HELPERS ------------------- #
    @staticmethod
    def _read_second_rating_block(csv_path: str) -> np.ndarray:
        """
        Internal helper: open a rating CSV, find the block after '%%%%%%',
        parse as two columns [Second, Rating], and return (N, 2).
        """
        from io import StringIO

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        start = None
        for i, ln in enumerate(lines):
            if "%%%%%%" in ln:
                start = i + 1
                break
        if start is None:
            raise ValueError(f"'%%%%%%' not found in file: {csv_path}")

        numeric_text = "".join(lines[start:])
        df = pd.read_csv(StringIO(numeric_text), header=None, names=["Second", "Rating"])
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        return df[["Second", "Rating"]].to_numpy(dtype=np.float32)


    # ------------------- RATING LOADER ------------------- #
    def load_ratings_tp_all_subjects(
        self,
        root_dir: str,
        target_file_pattern: str = "*-TP.csv",
        strict_same_length: bool = True,
        pad_value: float = np.nan,
    ):
        """
        Read TP-rating CSVs under root_dir/*/, each containing '%%%%%%' delimiter.
        Store (num_subjects, N, 2) in self.rating_cube and subjects in self.rating_subjects.
        """
        root = Path(root_dir)
        paths = sorted(root.glob(f"*/{target_file_pattern}"))
        if not paths:
            raise FileNotFoundError(f"No files matched: {root}/*/{target_file_pattern}")

        mats, subjects, lengths = [], [], []
        for p in paths:
            mat = self._read_second_rating_block(str(p))
            mats.append(mat)
            subjects.append(p.parent.name)
            lengths.append(mat.shape[0])

        if strict_same_length:
            N0 = lengths[0]
            if any(L != N0 for L in lengths):
                details = ", ".join(f"{s}:{L}" for s, L in zip(subjects, lengths))
                raise ValueError("Not the same length across subjects:\n" + details)
            arr = np.stack(mats, axis=0).astype(np.float32)
        else:
            maxN = max(lengths)
            arr = np.full((len(mats), maxN, 2), pad_value, dtype=np.float32)
            for i, mat in enumerate(mats):
                n = mat.shape[0]
                arr[i, :n, :] = mat

        self.rating_cube = arr
        self.rating_subjects = subjects

        if self.verbose:
            print("[load_ratings_tp_all_subjects] cube shape:", arr.shape)
            print("[load_ratings_tp_all_subjects] first 5 subjects:", subjects[:5])

    def load_atlas_info(
        self,
        schaefer_centroid_csv: str,
        out_csv: str | None = None,
        append_aal: bool = True,
        aal_text: str | None = None,
    ):
        """
        Build an atlas table by:
          1) reading Schaefer-100 centroid CSV,
          2) parsing Hemisphere/Subnetwork/Region/Parcel index from ROI Name,
          3) optionally appending AAL ROIs (IDs 101..114) from text lines,
          4) saving to self.atlas_df (and optionally write to CSV).

        After this, call self.build_roi_groupings() to populate grouping dicts.
        """
        import re
        from pathlib import Path

        # --- 1) read Schaefer 100 ---
        df = pd.read_csv(schaefer_centroid_csv, skipinitialspace=True)
        labels = df[["ROI Label", "ROI Name"]].to_numpy()
        coords  = df[["R", "A", "S"]].to_numpy(dtype=float)

        # --- 2) parse ROI Name robustly ---
        split_rows = []   # [Hemisphere, Subnetwork, Region, Parcel index]
        valid_subnets = {'Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default'}

        for label_id, name in labels:
            parts = str(name).split('_')

            hemi = next((p for p in parts if p in ('LH','RH')), None)
            subn = next((p for p in parts if p in valid_subnets), None)
            par_idx = None
            if parts and parts[-1].isdigit():
                par_idx = int(parts[-1])

            region = None
            if subn is not None:
                try:
                    i = parts.index(subn)
                    if i + 1 < len(parts):
                        cand = parts[i + 1]
                        if cand not in ('LH','RH') and not cand.isdigit():
                            region = cand
                except ValueError:
                    pass

            if hemi is None:   hemi = 'LH'
            if subn is None:   subn = 'Unknown'
            if region is None: region = 'roi'
            if par_idx is None:
                try:
                    par_idx = int(label_id)
                except Exception:
                    par_idx = -1

            split_rows.append([hemi, subn, region, par_idx])

        df_1 = pd.DataFrame(split_rows, columns=['Hemisphere','Subnetwork','Region','Parcel index'])
        df_2 = pd.DataFrame(np.column_stack([labels[:, 0], coords]),
                            columns=['ROI ID','R','A','S'])
        df_2['ROI ID'] = df_2['ROI ID'].astype(int)
        df_2[['R','A','S']] = df_2[['R','A','S']].astype(float)

        atlas = pd.concat([df_1, df_2], axis=1).reset_index(drop=True)

        # --- 3) optionally append AAL 101..114 ---
        if append_aal:
            if aal_text is None:
                aal_text = """
                            ROI # 101: AAL# 37 Hippocampus_L 4101
                            ROI # 102: AAL# 38 Hippocampus_R 4102
                            ROI # 103: AAL# 39 ParaHippocampal_L 4111
                            ROI # 104: AAL# 40 ParaHippocampal_R 4112
                            ROI # 105: AAL# 41 Amygdala_L 4201
                            ROI # 106: AAL# 42 Amygdala_R 4202
                            ROI # 107: AAL# 71 Caudate_L 7001
                            ROI # 108: AAL# 72 Caudate_R 7002
                            ROI # 109: AAL# 73 Putamen_L 7011
                            ROI # 110: AAL# 74 Putamen_R 7012
                            ROI # 111: AAL# 75 Pallidum_L 7021
                            ROI # 112: AAL# 76 Pallidum_R 7022
                            ROI # 113: AAL# 77 Thalamus_L 7101
                            ROI # 114: AAL# 78 Thalamus_R 7102
                                            """.strip()

            pat = re.compile(r"ROI\s*#\s*(\d+):\s*AAL#\s*(\d+)\s+([A-Za-z]+)_(L|R)\s+(\d+)")
            rows1, rows2 = [], []
            for line in aal_text.splitlines():
                m = pat.search(line.strip())
                if not m:
                    continue
                roi_num, _aal_idx, region, lr, _aal_code = m.groups()
                hemi = "LH" if lr == "L" else "RH"
                rows1.append({
                    "Hemisphere": hemi,
                    "Subnetwork": "AAL",
                    "Region": region,
                    "Parcel index": pd.NA,
                })
                rows2.append({
                    "ROI ID": int(roi_num),
                    "R": np.nan, "A": np.nan, "S": np.nan,
                })

            df_1_new = pd.DataFrame(rows1)
            df_2_new = pd.DataFrame(rows2).astype({"ROI ID": int, "R": float, "A": float, "S": float})
            df_1_new['Parcel index'] = df_1_new['Parcel index'].astype('Int64')

            atlas = pd.concat([atlas, pd.concat([df_1_new.reset_index(drop=True),
                                                 df_2_new.reset_index(drop=True)], axis=1)],
                              ignore_index=True)

        # --- 4) add 'Atlas' label & reorder columns ---
        def which_atlas(roi_id: int) -> str:
            return "Schaefer2018_7Networks" if int(roi_id) <= 100 else "AAL"
        atlas['Atlas'] = atlas['ROI ID'].astype(int).apply(which_atlas)

        cols = ['Atlas','Hemisphere','Subnetwork','Region','Parcel index','ROI ID','R','A','S']
        atlas = atlas.reindex(columns=cols)

        self.atlas_df = atlas

        if out_csv:
            out_path = Path(out_csv)
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                atlas.to_csv(out_path, index=False)
                if self.verbose:
                    print(f"[load_atlas_info] saved atlas CSV → {out_path}")
            else:
                if self.verbose:
                    print(f"[load_atlas_info] already exists, not saving → {out_path}")


        if self.verbose:
            print("[load_atlas_info] atlas shape:", atlas.shape)


    def build_roi_groupings(self):
        """
        Build two grouping dicts from self.atlas_df:
          - self.roi_by_network_hemisphere: {Subnetwork: {LH/RH: [ROI IDs]}}
          - self.roi_by_atlas_network_hemi: {Atlas: {Subnetwork: {LH/RH: [ROI IDs]}}}
        """
        if self.atlas_df is None:
            raise RuntimeError("atlas_df is None. Call load_atlas_info(...) first.")

        df_min = self.atlas_df[['Atlas','Hemisphere','Subnetwork','ROI ID']].copy()

        # Subnetwork × Hemisphere
        d1 = {}
        grouped1 = df_min.groupby(['Subnetwork','Hemisphere'])['ROI ID'].apply(list)
        for (subnet, hemi), roi_list in grouped1.items():
            d1.setdefault(subnet, {})[hemi] = roi_list
        self.roi_by_network_hemisphere = d1

        # Atlas × Subnetwork × Hemisphere
        d2 = {}
        grouped2 = df_min.groupby(['Atlas','Subnetwork','Hemisphere'])['ROI ID'].apply(list)
        for (atlas, subnet, hemi), roi_list in grouped2.items():
            d2.setdefault(atlas, {}).setdefault(subnet, {})[hemi] = roi_list
        self.roi_by_atlas_network_hemi = d2

        if self.verbose:
            print("[build_roi_groupings] built:",
                  f"Subnetwork×Hemisphere groups={len(d1)};",
                  f"Atlas×Subnetwork×Hemisphere atlases={len(d2)}")


    def multi_kernel_encoding(
        self,
        features_dict,            # dict: {feature_name: X (T, d_k)}
        feature_order,            # list[str]: ordered list of feature names
        roi_data=None,            # optional; if None, will use self.roi_all (shape expected: [S, ROI, T])
        n_delays=5,               # placeholder; add_delays is commented below
        outer_folds=5,
        num_alphas=30,
        output_dir="./mke_out",
        prefix="mke",
        save_predictions=False,
    ):
        """
        Multi-kernel ridge encoding with per-feature R² decomposition.
        Nothing is returned; results are written to disk.

        Saved outputs:
            - {prefix}_results_r2.npy          : overall R² across ROIs (shape = [S, ROI])
            - {prefix}_feature_r2s.npy         : per-feature R² values (shape = [S, F, ROI])
            - {prefix}_summary.csv             : readable summary (per subject + per feature means)
            - {prefix}_meta.json               : metadata (parameters, shapes, feature list)
            - {prefix}_pred_subj{idx}.npz      : (optional) per-subject predictions if save_predictions=True
        """


        # External dependencies expected in your codebase:
        # from yourlib import Kernelizer, ColumnKernelizer, MultipleKernelRidgeCV, r2_score_split
        # If these live elsewhere, adjust the import path accordingly.

        # ------------------------- Resolve ROI data ------------------------- #
        # If roi_data wasn't provided, try using self.roi_all.
        if roi_data is None:
            if self.roi_all is None:
                raise ValueError("roi_data is None and self.roi_all is not set.")
            roi_data = self.roi_all
        # roi_data expected shape: (n_subjects, n_rois, T)

        os.makedirs(output_dir, exist_ok=True)

        # Log-spaced alpha values for ridge regularization
        alphas = np.logspace(-5, 5, num_alphas)
        n_subjects = int(roi_data.shape[0])
        n_rois     = int(roi_data.shape[1])
        n_features_total = len(feature_order)

        # Allocate result containers
        results_r2  = np.zeros((n_subjects, n_rois), dtype=np.float32)                  # overall R²
        feature_r2s = np.zeros((n_subjects, n_features_total, n_rois), dtype=np.float32) # per-feature R²

        if self.verbose:
            print(f"[MKE] subjects={n_subjects}, rois={n_rois}, features={n_features_total}")

        # ========================== Subject loop =========================== #
        for subj in range(n_subjects):
            if self.verbose:
                print(f"=== Subject {subj + 1}/{n_subjects} ===")

            # Y: time-by-voxel/ROI matrix; z-score per ROI across time
            Y = roi_data[subj].T  # shape [T, ROI]
            Y = stats.zscore(Y, axis=0, nan_policy="omit")

            # ------------------------------------------------------
            # Step 1: Delay (optional) and standardize each feature
            # ------------------------------------------------------
            delayed_features, feature_dims = [], []
            for f in feature_order:
                X = features_dict[f]
                # If temporal delays are required, uncomment and provide add_delays:
                # X = add_delays(X, n_delays=n_delays)   # -> [T, d_k * (n_delays+1)]
                X = StandardScaler().fit_transform(X)    # z-score per column
                delayed_features.append(X)
                feature_dims.append(X.shape[1])

            # ------------------------------------------------------
            # Step 2: Concatenate feature blocks along columns
            # ------------------------------------------------------
            X_all = np.concatenate(delayed_features, axis=1)  # [T, sum(d_k)]
            if self.verbose:
                print(f"X_all shape = {X_all.shape} (sum of {len(feature_order)} features)")

            # ------------------------------------------------------
            # Step 3: Build index slices per feature block
            # ------------------------------------------------------
            start_end = np.concatenate([[0], np.cumsum(feature_dims)])
            slices = [slice(start, end) for start, end in zip(start_end[:-1], start_end[1:])]

            # ------------------------------------------------------
            # Step 4: Per-feature kernelizers (linear kernels here)
            # ------------------------------------------------------
            kernel_tuples = [
                (name, Kernelizer(kernel="linear"), sl)
                for name, sl in zip(feature_order, slices)
            ]
            column_kernelizer = ColumnKernelizer(kernel_tuples)

            # ------------------------------------------------------
            # Step 5: Temporal group K-fold splits (chunked)
            # ------------------------------------------------------
            n_samples = Y.shape[0]
            chunk_len = 24
            n_chunks  = int(n_samples / chunk_len)
            groups    = [i for i in range(n_chunks) for _ in range(chunk_len)]
            if len(groups) < n_samples:  # tail samples in a final (possibly shorter) chunk
                groups.extend([n_chunks] * (n_samples - len(groups)))

            cv_outer = GroupKFold(n_splits=outer_folds)
            splits = cv_outer.split(np.arange(n_samples), groups=groups)

            # ------------------------------------------------------
            # Step 6: Multi-kernel ridge with random-search solver
            # ------------------------------------------------------
            mkr = MultipleKernelRidgeCV(
                kernels="precomputed",
                solver="random_search",
                solver_params=dict(
                    alphas=alphas,
                    local_alpha=True,
                    n_iter=50,
                    diagonalize_method="svd",
                    # random_state=0,  # uncomment for reproducibility if supported
                ),
                cv=splits
            )
            pipeline = make_pipeline(column_kernelizer, mkr)

            # ------------------------------------------------------
            # Step 7: Fit model and compute R²
            # ------------------------------------------------------
            pipeline.fit(X_all, Y)
            scores = pipeline.score(X_all, Y)                 # iterable per ROI
            results_r2[subj] = np.array(scores, dtype=np.float32)

            # Per-feature R² via split predictions
            Y_pred_split = pipeline.predict(X_all, split=True)  # list/array per feature
            split_r2 = r2_score_split(Y, Y_pred_split)          # -> shape (F, ROI)
            feature_r2s[subj] = np.array(split_r2, dtype=np.float32)

            if self.verbose:
                print(f"  mean R²_total = {np.nanmean(results_r2[subj]):.4f}")
                for i, f in enumerate(feature_order):
                    print(f"    {f:<15s} → mean R² = {np.nanmean(feature_r2s[subj, i]):.4f}")

            # Optionally save per-subject predictions (can be large on disk)
            if save_predictions:
                np.savez_compressed(
                    os.path.join(output_dir, f"{prefix}_pred_subj{subj:02d}.npz"),
                    Y_true=Y.astype(np.float32),
                    Y_pred_split=Y_pred_split,                   # if it's a list, it's saved as object array by npz
                    feature_order=np.array(feature_order, dtype=object),
                )

        # ============================ Save to disk ============================ #
        np.save(os.path.join(output_dir, f"{prefix}_results_r2.npy"), results_r2)
        np.save(os.path.join(output_dir, f"{prefix}_feature_r2s.npy"), feature_r2s)

        # Human-readable summary CSV (subject means + per-feature subject means)
        df_summary = pd.DataFrame({
            "subject_idx": np.arange(n_subjects),
            "mean_total_R2": np.nanmean(results_r2, axis=1),
        })
        for i, fname in enumerate(feature_order):
            df_summary[f"feat_{fname}_mean_R2"] = np.nanmean(feature_r2s[:, i, :], axis=1)
        df_summary.to_csv(os.path.join(output_dir, f"{prefix}_summary.csv"), index=False)

        # Metadata JSON for reproducibility
        meta = dict(
            n_subjects=int(n_subjects),
            n_rois=int(n_rois),
            n_features_total=int(n_features_total),
            n_delays=int(n_delays),
            outer_folds=int(outer_folds),
            num_alphas=int(num_alphas),
            chunk_len=int(chunk_len),
            feature_order=list(map(str, feature_order)),
            results_r2_shape=list(map(int, results_r2.shape)),
            feature_r2s_shape=list(map(int, feature_r2s.shape)),
            overall_mean_R2=float(np.nanmean(results_r2)),
            save_predictions=bool(save_predictions),
        )
        with open(os.path.join(output_dir, f"{prefix}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"[MKE] Finished. Results saved to: {os.path.abspath(output_dir)}")

    def load_features(self, root_dir: str = ".", pattern_map: dict | None = None):
        """
        Placeholder for feature loading.
        Later this function will load all feature CSVs into self.feature_dict.
        """
        self.feature_dict = {}
        self.feature_names = []
        if self.verbose:
            print(f"[load_features] Placeholder called. root_dir={root_dir}, pattern_map={pattern_map}")


# ------------------- MAIN TEST ------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./Movie_ROI")
    parser.add_argument("--roi_folder", type=str, default="movie_tp")
    parser.add_argument("--roi_pattern", type=str, default="sub-*_ROI114.csv")

    parser.add_argument("--rating_root", type=str, default="Rating_ToM")
    parser.add_argument("--rating_pattern", type=str, default="*-TP.csv")

    parser.add_argument("--atlas_centroid_csv", type=str,
                        default="./atlas/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv")
    parser.add_argument("--atlas_out_csv", type=str,
                        default="./atlas/Schaefer2018_114Parcels_7Networks_order_info.csv")
 
    args = parser.parse_args()

    model = EncodingModel(data_dir=args.data_dir)

    roi_folder_path = str(Path(args.data_dir) / args.roi_folder)
    rating_root_path = str(Path(args.data_dir) / args.rating_root)

    # Example usage
    model.load_roi_cube(
        folder=roi_folder_path,
        pattern=args.roi_pattern
    )
    model.load_ratings_tp_all_subjects(
        root_dir=rating_root_path,
        target_file_pattern=args.rating_pattern
    )

    model.load_atlas_info(
        schaefer_centroid_csv=args.atlas_centroid_csv,
        out_csv=args.atlas_out_csv,
    )
    model.build_roi_groupings()

    print("ROI cube:", model.roi_all.shape)
    print("Rating cube:", model.rating_cube.shape)
    print("Atlas table:", model.atlas_df.shape)
    d = model.roi_by_network_hemisphere or {}
    if "Default" in d:
        print("Default-LH:", d["Default"].get("LH"))
        print("Default-RH:", d["Default"].get("RH"))

    model.load_features(root_dir="./Movie_ROI/features")

    model.multi_kernel_encoding(
        features_dict=model.feature_dict,
        feature_order=list(model.feature_names),
        roi_data=None,                 # None -> use model.roi_all
        outer_folds=5,
        num_alphas=30,
        output_dir="./mke_out",
        prefix="tp_movie",
        save_predictions=False,
    )

if __name__ == "__main__":
    main()
