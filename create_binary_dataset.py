import os

def extract_binary_data(input_list, output_list):
    # Mapping old labels to new binary labels
    # 0 = Cargo -> New: 0
    # 3 = Tug   -> New: 1
    target_classes = {'0': '0', '3': '1'}
    count_0 = 0
    count_1 = 0
    
    os.makedirs(os.path.dirname(output_list), exist_ok=True)
    
    with open(input_list, 'r', encoding='utf-8') as f_in, \
         open(output_list, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                path, label = parts
                if label in target_classes:
                    new_label = target_classes[label]
                    f_out.write(f"{path}\t{new_label}\n")
                    if new_label == '0':
                        count_0 += 1
                    else:
                        count_1 += 1
                        
    print(f"Processed {input_list}:")
    print(f"  Class 0 (Cargo): {count_0}")
    print(f"  Class 1 (Tug):   {count_1}")
    print(f"  Total: {count_0 + count_1} saved to {output_list}\n")

if __name__ == "__main__":
    train_in = 'data/train_list.txt'
    train_out = 'data/binary_train_list.txt'
    
    test_in = 'data/test_list.txt'
    test_out = 'data/binary_test_list.txt'
    
    extract_binary_data(train_in, train_out)
    extract_binary_data(test_in, test_out)
