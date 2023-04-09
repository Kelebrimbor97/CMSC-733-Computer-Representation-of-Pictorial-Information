import numpy as np

# filename = 'Data/matching1.txt'
def find_matches(filename, image_no, flag = False):
    correspondence_list_1 = []
    correspondence_list_2 = []
    row=[]
    with open(filename) as f:
        line_no = 0
        for line in f:
            line_no = line_no + 1
            # print(line_no)
            if line_no == 1:
                continue
            
            l = line.split()
            for i in range(int(l[0]) - 1):
                if int(l[6+ 3*i]) == image_no:
                    if flag:
                        row.append(line_no)
                    correspondence_list_1.append([float(l[4]), float(l[5])])
                    correspondence_list_2.append([float(l[6+3*i + 1]), float(l[6+3*i + 2])])
    return [np.array(correspondence_list_1), np.array(correspondence_list_2)], np.array(row)

def find_common_points(filename, image_no, row_indices):
    common_points=[]
    common_indices = []
    file = open(filename)
    content = file.readlines()
    for i in row_indices:
        line = content[i-1]
        l = line.split()
        for j in range(int(l[0]) - 1):
            if int(l[6+ 3*j]) == image_no:
                common_indices.append(i)
                common_points.append([float(l[6+3*j + 1]), float(l[6+3*j + 2])])
    return np.array(common_points), np.array(common_indices)


def get_correspondence():
    I12, rows= find_matches('Data/matching1.txt', 2, True)
    I13,_= find_matches('Data/matching1.txt', 3)
    I14,_= find_matches('Data/matching1.txt', 4)
    I15,_= find_matches('Data/matching1.txt', 5)
    I16,_= find_matches('Data/matching1.txt', 6)
    I23,_= find_matches('Data/matching2.txt', 3)
    I24,_= find_matches('Data/matching2.txt', 4)
    I25,_= find_matches('Data/matching2.txt', 5)
    I26,_= find_matches('Data/matching2.txt', 6)
    I34,_= find_matches('Data/matching3.txt', 4)
    I35,_= find_matches('Data/matching3.txt', 5)
    I36,_= find_matches('Data/matching3.txt', 6)
    I45,_= find_matches('Data/matching4.txt', 5)
    I46,_= find_matches('Data/matching4.txt', 6)
    I56,_= find_matches('Data/matching5.txt', 6)

    return [I12, I13, I14, I15, I16, I23, I24, I25, I26, I34, I35, I36, I45, I46, I56], rows


