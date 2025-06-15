resource_dir = "data/resources/"

with open(resource_dir + "auteurs-fr.txt", "r") as f:
    lines = f.readlines()
    # each line should contain a book id (int)
    # check if line is a number
    # if so, add it to the list of book ids
    # if not, skip it
    book_ids = []
    for line in lines:
        if line.strip().isdigit():
            book_ids.append(int(line.strip()))

with open(resource_dir + "dumas.txt", "r") as f:
    lines = f.readlines()
    # each line should contain a book id (int)
    # check if line is a number
    # if so, add it to the list of book ids
    # if not, skip it
    dumas_ids = []
    for line in lines:
        if line.strip().isdigit():
            dumas_ids.append(int(line.strip()))

for d in dumas_ids:
    if d not in book_ids:
        print(f"{d} not in book_ids")