import numpy as np
import matplotlib.pyplot as plt

def detect_rising_edges(signal, threshold=0.5, hysteresis=0.1):
    """Detect rising edges with hysteresis."""
    edges = []
    in_high = False
    for i in range(1, len(signal)):
        if not in_high and signal[i] > threshold + hysteresis:
            edges.append(i)
            in_high = True
        elif in_high and signal[i] < threshold - hysteresis:
            in_high = False
    return np.array(edges)

def sinc_interpolation(signal, position, window_size=8):
    """Sinc interpolation for non-integer positions."""
    idx = int(np.floor(position))
    x = np.arange(idx - window_size//2, idx + window_size//2)
    # Handle boundaries
    x = x[(x >= 0) & (x < len(signal))]
    sinc_kernel = np.sinc(x - position)
    return np.sum(signal[x] * sinc_kernel) / np.sum(sinc_kernel)

def align_trace(ref_edges, target_edges, target_trace, trace_len):
    """Align target_trace to ref_edges using target_edges."""
    aligned = np.zeros(trace_len)
    n_segments = min(len(ref_edges), len(target_edges))
    for seg in range(n_segments - 1):
        ref_start, ref_end = ref_edges[seg], ref_edges[seg+1]
        tgt_start, tgt_end = target_edges[seg], target_edges[seg+1]
        for i, ref_pos in enumerate(range(ref_start, ref_end)):
            # Linear mapping from reference to target
            alpha = i / max(1, ref_end - ref_start)
            tgt_pos = tgt_start + alpha * (tgt_end - tgt_start)
            if 0 <= tgt_pos < len(target_trace) - 1:
                aligned[ref_pos] = sinc_interpolation(target_trace, tgt_pos)
    return aligned

def main(key_byte, n_traces, n_samples, threshold):
    # Load a small subset for visualization
    traces = np.loadtxt(f'trace{key_byte}.txt', dtype=np.float32, max_rows=n_traces, usecols=range(n_samples))
    clocks = np.loadtxt(f'clock{key_byte}.txt', dtype=np.float32, max_rows=n_traces, usecols=range(n_samples))
    
    # Normalize for visualization
    traces = (traces - traces.mean(axis=1, keepdims=True)) / traces.std(axis=1, keepdims=True)
    clocks = (clocks - clocks.mean(axis=1, keepdims=True)) / clocks.std(axis=1, keepdims=True)
    
    # Detect rising edges in reference
    ref_edges = detect_rising_edges(clocks[0], threshold)
    
    # Align all traces
    aligned_traces = []
    for i in range(n_traces):
        target_edges = detect_rising_edges(clocks[i], threshold)
        aligned = align_trace(ref_edges, target_edges, traces[i], n_samples)
        aligned_traces.append(aligned)
        print(f"Aligned trace {i}")
    
    aligned_traces = np.array(aligned_traces)
    np.savetxt(f'aligned_traces_byte{key_byte}.txt', aligned_traces)

if __name__ == "__main__":
    for key_byte in range(16):
        print(f"Processing byte {key_byte}...")
        main(key_byte=key_byte, n_traces=150, n_samples=50000, threshold=0.5)