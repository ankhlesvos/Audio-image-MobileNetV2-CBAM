import os

def create_merged_data(input_list, output_list, balance=False):
    # Mapping old labels to merged labels:
    # 0 = Cargo       -> New: 0 (Cargo_Tug)
    # 1 = Passenger   -> New: 1
    # 2 = Tanker      -> New: 2
    # 3 = Tug         -> New: 0 (Cargo_Tug)
    
    label_map = {
        '0': '0',
        '1': '1',
        '2': '2',
        '3': '0'
    }
    
    # Store lines by new label
    class_lines = {'0': [], '1': [], '2': []}
    
    os.makedirs(os.path.dirname(output_list), exist_ok=True)
    
    with open(input_list, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                path, label = parts
                if label in label_map:
                    new_label = label_map[label]
                    class_lines[new_label].append(f"{path}\t{new_label}\n")
    
    balanced_lines = []
    
    if balance:
        import random
        random.seed(42)
        max_count = max(len(lines) for lines in class_lines.values())
        
        for label, lines in class_lines.items():
            if len(lines) == 0:
                continue
            # Oversample: keep all original, then randomly sample the difference
            diff = max_count - len(lines)
            oversampled = lines + random.choices(lines, k=diff)
            balanced_lines.extend(oversampled)
        random.shuffle(balanced_lines)
        
        print(f"Processed {input_list} (Balanced via Oversampling):")
        for label in ['0', '1', '2']:
            print(f"  Class {label} (Original: {len(class_lines[label])}) -> Balanced: {max_count}")
    else:
        for label, lines in class_lines.items():
            balanced_lines.extend(lines)
        
        print(f"Processed {input_list} (Natural Distribution):")
        for label in ['0', '1', '2']:
            print(f"  Class {label}: {len(class_lines[label])}")
        
    with open(output_list, 'w', encoding='utf-8') as f_out:
        f_out.writelines(balanced_lines)
                        
    print(f"  Total: {len(balanced_lines)} saved to {output_list}\n")
    


if __name__ == "__main__":
    train_in = 'data/train_list.txt'
    train_out = 'data/merged_train_list.txt'
    
    test_in = 'data/test_list.txt'
    test_out = 'data/merged_test_list.txt'
    
    create_merged_data(train_in, train_out, balance=True)
    create_merged_data(test_in, test_out, balance=False)
