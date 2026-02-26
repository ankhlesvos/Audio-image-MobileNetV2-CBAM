import os, glob

BYTES_PER_SEC = 32000 * 2  # 32kHz 16-bit mono

print("="*65)
print(f"{'Class':<16} {'Files':>6} {'TotalMB':>9} {'TotalHrs':>10} {'AvgMin':>8}")
print("-"*65)

class_data = {}
for cls in ['Cargo', 'Passengership', 'Tanker', 'Tug']:
    files = glob.glob(f'DeepShip-main/{cls}/*.wav')
    sizes = [os.path.getsize(f) for f in files]
    total_bytes = sum(sizes)
    total_secs = total_bytes / BYTES_PER_SEC
    avg_min = total_secs / len(files) / 60 if files else 0
    class_data[cls] = {'n': len(files), 'secs': total_secs}
    print(f"{cls:<16} {len(files):>6} {total_bytes/1e6:>9.1f} {total_secs/3600:>10.2f} {avg_min:>8.1f}")

    # Per-file breakdown
    file_pairs = sorted(zip(sizes, files))
    short = [(s, f) for s, f in file_pairs if s/BYTES_PER_SEC < 60]
    if short:
        print(f"  -> {len(short)} files SHORTER than 1 min: {[round(s/BYTES_PER_SEC) for s,f in short[:5]]}s ...")

print()
print("Merged class 0 = Cargo + Tug:")
merged_secs = class_data['Cargo']['secs'] + class_data['Tug']['secs']
print(f"  Total {merged_secs/3600:.2f}h  ({class_data['Cargo']['n']+class_data['Tug']['n']} files)")

print()
SEGMENT = 5
STRIDE  = 0.5  # overlap=0.9

print(f"=== Estimated TRAIN segments (5s, stride=0.5s, 80% train split) ===")
mappings = {
    'Cargo+Tug': class_data['Cargo']['secs'] + class_data['Tug']['secs'],
    'Passengership': class_data['Passengership']['secs'],
    'Tanker': class_data['Tanker']['secs'],
}
for cls, total_secs in mappings.items():
    train_secs = total_secs * 0.8
    n_files = class_data.get(cls.split('+')[0], {}).get('n', 0)
    est_segs = int(train_secs / STRIDE)  # upper bound
    print(f"  {cls:<16}: {train_secs/3600:.2f}h -> ~{est_segs} segs (upper bound, before VAD)")
