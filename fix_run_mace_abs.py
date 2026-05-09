import json

nb_file = r"g:\Mi unidad\gnn-material-science\run-MACE.ipynb"
with open(nb_file, "r", encoding="utf-8") as f:
    data = json.load(f)

modified = False
for cell in data.get("cells", []):
    if cell["cell_type"] == "code":
        for i in range(len(cell["source"])-1, -1, -1):
            line = cell["source"][i]
            if "!pip install -r requirements.txt" in line:
                cell["source"].insert(i+1, "    !pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12\n")
                modified = True
                break
        if modified:
            break

if modified:
    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("No changes made.")
