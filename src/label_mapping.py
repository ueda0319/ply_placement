import csv

def load_label_mapping(category_type: str="ShapeNetCore55", id_type: str="nyu40id"):
    with open('labels/scannetv2.tsv', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')

        header = next(reader)
        category_col = header.index(category_type)
        id_col = header.index(id_type)

        label_mapping = {}

        # 残りの行を一行ずつ読み込み
        for cols in reader:
            if cols[category_col] == "" or cols[id_col] == "":
                continue
            label_mapping[cols[category_col]] = int(cols[id_col])
        
        return label_mapping


if __name__ == "__main__":
    label_mapping = load_label_mapping()
    print(label_mapping)
