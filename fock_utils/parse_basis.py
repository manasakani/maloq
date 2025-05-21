def parse_basis_set(file_path):
    basis_dict = {}
    current_element = None
    basis_list = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.isalpha():
                if current_element is not None:
                    # Save the previous element's basis list
                    basis_dict[current_element] = basis_list
                # Start a new element
                current_element = line
                basis_list = []
            elif line.startswith(('S', 'P', 'D', 'F', 'G')):
                basis_type = line[0]
                if basis_type == 'S':
                    basis_list.append(0)
                elif basis_type == 'P':
                    basis_list.append(1)
                elif basis_type == 'D':
                    basis_list.append(2)
                elif basis_type == 'F':
                    basis_list.append(3)
                elif basis_type == 'G':
                    basis_list.append(4)

        if current_element is not None:
            basis_dict[current_element] = basis_list

    return basis_dict

file_path = 'def2-tzvpd.txt'
basis_set_dict = parse_basis_set(file_path)
print(basis_set_dict)