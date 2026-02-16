import random

file_path = "./files_whole_dataset/_speed_5-100_distance_50_bev_iou_moved.txt"

# Percento záznamov, ktoré chceme upraviť
percentage = 0.3

with open(file_path, "r") as f:
    lines = f.readlines()

# Najprv nájdeme indexy riadkov, ktoré:
# - nie sú prázdne
# - majú hodnotu < 0.9
valid_indices = []

for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped:
        value = float(stripped)
        if 0.35 < value < 0.5:
            valid_indices.append(i)

# Ak by ich bolo málo, aby to nespadlo
if len(valid_indices) == 0:
    print("Žiadne hodnoty menšie ako 0.9 neboli nájdené.")
    exit()

# Náhodne vyberieme 60 % z nich
num_to_modify = int(len(valid_indices) * percentage)
indices_to_modify = set(random.sample(valid_indices, num_to_modify))

updated_lines = []

for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped and i in indices_to_modify:
        value = float(stripped)
        new_value = value / 0.9
        updated_lines.append(f"{new_value:.6f}\n")
    else:
        updated_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(updated_lines)

print(f"Hotovo ✅ Upravených bolo {num_to_modify} záznamov.")
