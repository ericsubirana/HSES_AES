import numpy as np
from joblib import Parallel, delayed

def detect_rising_edges(signal, threshold=0.5, hysteresis=0.1):
    edges = []
    in_high = False
    for i in range(1, len(signal)):
        if not in_high and signal[i] > threshold + hysteresis:
            edges.append(i)
            in_high = True
        elif in_high and signal[i] < threshold - hysteresis:
            in_high = False
    return np.array(edges)

def sinc_interpolation(signal, position, window_size=4):
    idx = int(np.floor(position))
    x = np.arange(idx - window_size//2, idx + window_size//2)
    x = x[(x >= 0) & (x < len(signal))]
    sinc_kernel = np.sinc(x - position)
    denom = np.sum(sinc_kernel)
    if denom == 0:
        return 0
    return np.sum(signal[x] * sinc_kernel) / denom

def align_trace(ref_edges, target_edges, target_trace, trace_len, window_size=4):
    aligned = np.zeros(trace_len, dtype=np.float32)
    n_segments = min(len(ref_edges), len(target_edges))
    for seg in range(n_segments - 1):
        ref_start, ref_end = ref_edges[seg], ref_edges[seg+1]
        tgt_start, tgt_end = target_edges[seg], target_edges[seg+1]
        seg_len = ref_end - ref_start
        tgt_len = tgt_end - tgt_start
        if seg_len == 0 or tgt_len == 0:
            continue
        alpha = np.linspace(0, 1, seg_len, endpoint=False)
        tgt_positions = tgt_start + alpha * tgt_len
        for i, ref_pos in enumerate(range(ref_start, ref_end)):
            tgt_pos = tgt_positions[i]
            if 0 <= tgt_pos < len(target_trace) - 1:
                aligned[ref_pos] = sinc_interpolation(target_trace, tgt_pos, window_size)
    return aligned

def main(key_byte, n_traces, n_samples, threshold):
    traces = np.loadtxt(f'trace{key_byte}.txt', dtype=np.float32, max_rows=n_traces, usecols=range(n_samples))
    clocks = np.loadtxt(f'clock{key_byte}.txt', dtype=np.float32, max_rows=n_traces, usecols=range(n_samples))
    traces = (traces - traces.mean(axis=1, keepdims=True)) / traces.std(axis=1, keepdims=True)
    clocks = (clocks - clocks.mean(axis=1, keepdims=True)) / clocks.std(axis=1, keepdims=True)
    ref_edges = detect_rising_edges(clocks[0], threshold)

    # Parallel alignment
    def align_one(i):
        target_edges = detect_rising_edges(clocks[i], threshold)
        return align_trace(ref_edges, target_edges, traces[i], n_samples, window_size=4)

    aligned_traces = Parallel(n_jobs=-1, prefer="threads")(
        delayed(align_one)(i) for i in range(n_traces)
    )
    aligned_traces = np.array(aligned_traces, dtype=np.float32)
    np.savetxt(f'aligned_traces_byte{key_byte}.txt', aligned_traces)

if __name__ == "__main__":
    for key_byte in range(16):
        print(f"Processing byte {key_byte}...")
        main(key_byte=key_byte, n_traces=150, n_samples=50000, threshold=0.5)
