from imports import *

layers = ["conv1","conv2","conv3","conv4","conv5","fc6","fc7"]

def resample_to_tr(data, t, start_sec=8.0, TR=0.8):

    assert len(t) == len(data), "t and data length not match"
    t = np.asarray(t)
    data = np.asarray(data)

    t_samples = np.arange(start_sec, t[-1] + 1e-9, TR)

    idx = np.searchsorted(t, t_samples, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 1)

    r = pd.Series(data).ffill().bfill().to_numpy()

    return r[idx]

def sample_tom_ratings(csv_path, start_sec=8.0, TR=0.8,
                       time_col="timestamp_sec", rating_col="tom_rating"):
    df = pd.read_csv(csv_path)
    t = df[time_col].to_numpy(dtype=float)
    r = df[rating_col].to_numpy(dtype=float)

    return resample_to_tr(r, t, start_sec, TR).tolist()


def ratings_per_TR(cube, TR=0.8, start_sec=None, end_sec=None):
    t = np.nanmean(cube[:, :, 0], axis=0)      # (812,)
    r = np.nanmean(cube[:, :, 1], axis=0)      # (812,)

    if start_sec is None:
        start_sec = float(t[0])
    if end_sec is None:
        end_sec = float(t[-1])

    t_samples = np.arange(start_sec, end_sec + 1e-9, TR)
    idx = np.searchsorted(t, t_samples, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 1)

    return r[idx].tolist() 

def load_four_groups(csv_path):

    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    t = df["timestamp_sec"].to_numpy(dtype=float)
    
    start_desc = cols.index("desc_emb_0")
    start_kw = cols.index("kw_emb_0")
    idx_num = cols.index("num_people")

    desc = df.iloc[:, start_desc:start_kw].to_numpy()
    kw = df.iloc[:, start_kw:idx_num].to_numpy()

    return desc, kw, t

def resample_embeddings_to_tr(data, t, start_sec=8.0, TR=0.8, method="mean"):
    """
        data : np.ndarray, shape (N, D)
        t    : np.ndarray, shape (N,) 
        start_sec : float             
        TR   : float                  
        method : {"nearest","mean"}    
    return:
        data_tr : np.ndarray, shape (T, D)
    """
    t = np.asarray(t, dtype=float)
    order = np.argsort(t)
    t = t[order]
    data = np.asarray(data)[order]

    keep = t >= start_sec
    t = t[keep]
    data = data[keep]

    t_end = float(t[-1])

    if method == "nearest":
        tr_times = np.arange(start_sec, t_end + 1e-9, TR)
        idx = np.searchsorted(t, tr_times, side="right") - 1
        idx = np.clip(idx, 0, len(t) - 1)

        data_filled = pd.DataFrame(data).ffill().bfill().to_numpy()
        data_tr = data_filled[idx]

    elif method == "mean":
        edges = np.arange(start_sec, t_end + TR, TR)   
        bins = np.digitize(t, edges) - 1             
        T = len(edges) - 1
        D = data.shape[1]
        out = np.empty((T, D), dtype=float); out[:] = np.nan
        for k in range(T):
            m = (bins == k)
            if np.any(m):
                out[k] = data[m].mean(axis=0)

        data_tr = pd.DataFrame(out).ffill().bfill().to_numpy()
    else:
        raise ValueError("method must be 'nearest' or 'mean'.")

    return data_tr


def load_first7_layers(csv_path):

    cols = pd.read_csv(csv_path, nrows=0).columns
    pat = re.compile(r'^(conv_?([1-5])|fc([6-8]))_')
    layer_cols = {l: [] for l in layers}

    for c in cols:
        m = pat.match(c)
        if not m:
            continue
        if m.group(2):   # conv
            layer = f"conv{m.group(2)}"
        else:            # fc
            layer = f"fc{m.group(3)}"
        if layer in layer_cols:
            layer_cols[layer].append(c)

    usecols = [c for L in layers for c in layer_cols[L]]
    df = pd.read_csv(csv_path, usecols=usecols)

    arrays = {}
    for L in layers:
        arr = df[layer_cols[L]].to_numpy()
        arrays[L] = arr
        print(f"{L}: {arr.shape}")  

    return arrays