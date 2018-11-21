
UNACC = 3
ACC = 1
GOOD = 2
VG = 4

def getNewClass(oldClass):
    if oldClass == UNACC:
        return 'unacceptable'
    if oldClass == ACC:
        return 'acceptable'
    if oldClass == GOOD:
        return 'good'
    if oldClass == VG:
        return 'very_good'

with open('car-evaluation.arff','r') as original:
    with open('car-evaluation2.arff', 'w') as destino:
        for line in original:
            data = line.split(',')
            print(data[-1])
            data[-1] = getNewClass(int(data[-1]))
            destino.write(','.join(data) + '\n')