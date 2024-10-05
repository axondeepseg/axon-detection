import os

labels_base_path = '../data-yolo/labels/'
images_base_path = '../data-yolo/images/'

subdirs = ['train', 'val', 'test']

def normalize_labels():
    for subdir in subdirs:
        labels_path = os.path.join(labels_base_path, subdir)
        images_path = os.path.join(images_base_path, subdir)

        for label_file in os.listdir(labels_path):
            if label_file.endswith('.txt'):
                label_file_path = os.path.join(labels_path, label_file)

                image_file = label_file.replace('.txt', '.png') 
                image_file_path = os.path.join(images_path, image_file)

                if not os.path.exists(image_file_path):
                    print(f"Image file {image_file_path} not found for {label_file}")
                    continue

                with open(label_file_path, 'r') as f:
                    lines = f.readlines()

                normalized_lines = []
                for line in lines:
                    values = list(map(float, line.split()))
                    if len(values) < 5:
                        print(f"Invalid format in {label_file}")
                        continue

                    # Normalize the bounding box coordinates (assuming YOLO format)
                    values[1] = min(max(values[1], 0.0), 1.0)  # x center
                    values[2] = min(max(values[2], 0.0), 1.0)  # y center
                    values[3] = min(max(values[3], 0.0), 1.0)  # width
                    values[4] = min(max(values[4], 0.0), 1.0)  # height

                    normalized_lines.append(' '.join(map(str, values)) + '\n')

                with open(label_file_path, 'w') as f:
                    f.writelines(normalized_lines)

                print(f"Normalized {label_file}")
