import json, csv, os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

input_path = os.path.join(project_root, "data", "processed", "descriptions.jsonl")
output_dir = os.path.join(project_root, "data", "processed")

os.makedirs(output_dir, exist_ok=True)

# Open all three files at once and write in a single pass over the JSONL
conv_path  = os.path.join(output_dir, "convergence_labels.csv")
stab_path  = os.path.join(output_dir, "stability_labels.csv")

fieldnames = ["text", "label", "source", "n_atoms", "stopping_criterion", "file_path"]

with open(input_path) as f, \
     open(conv_path, "w", newline="") as conv_out, \
     open(stab_path, "w", newline="") as stab_out:

    conv_writer = csv.DictWriter(conv_out, fieldnames=fieldnames)
    stab_writer = csv.DictWriter(stab_out, fieldnames=fieldnames)
    conv_writer.writeheader()
    stab_writer.writeheader()

    conv_count = stab_count = 0

    for line in f:
        rec = json.loads(line)
        label_type = rec.get("label_type")

        if label_type == "convergence":
            conv_writer.writerow({
                "text":              rec["text"],
                "label":             1 if rec["label"] == "Converged" else 0,
                "source":            rec.get("folder", "unknown"),
                "n_atoms":           rec.get("n_atoms", ""),
                "stopping_criterion": rec.get("label", ""),
                "file_path":         rec.get("id", ""),
            })
            conv_count += 1

        elif label_type == "stability":
            stab_writer.writerow({
                "text":              rec["text"],
                "label":             1 if rec["label"] == "Stable" else 0,
                "source":            rec.get("folder", "unknown"),
                "n_atoms":           rec.get("n_atoms", ""),
                "stopping_criterion": rec.get("label", ""),
                "file_path":         rec.get("id", ""),
            })
            stab_count += 1

print(f"Done. Output → {output_dir}")
print(f"  convergence_labels.csv : {conv_count} rows")
print(f"  stability_labels.csv   : {stab_count} rows")