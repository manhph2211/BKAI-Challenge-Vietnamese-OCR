with open('vn_dictionary.txt','r') as f:
    new_lines = ''
    lines = f.readlines()
    for line in lines:
        if "##" in line:
            continue
        line = line.replace(" ","" )
        new_lines+=line

with open('custom.txt','w') as f:
    f.write(new_lines)
