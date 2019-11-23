# 词典构建及数据预处理
from util import util as use
from util import timingTool
import numpy as np
import xlrd
from mittens import GloVe
import numpy as np
import gc

tic = timingTool()
MATCH_ROW_NUMBER1 = {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,
                    "GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16}
# 共现矩阵的计算
def countCOOC(cooccurrence, window, coreIndex):
    for index in range(len(window)):
        if index == coreIndex:
            continue
        else:
            cooccurrence[window[coreIndex]][window[index]] = cooccurrence[window[coreIndex]][window[index]] + 1
    return cooccurrence

InputFile = xlrd.open_workbook(r'...\offtarget_data\Classification\off_data_twoset.xlsx')
#InputFile = xlrd.open_workbook(r'...\offtarget_data\Regression\off_data_twoset.xlsx')
outputList = "...\Data\Class\off_Glove.txt"
outputList1 = "...\Data\Class\hek293_off_Glove.txt"
outputList2 = "...\Data\Class\K562_off_Glove.txt"
sheet_hek293t = InputFile.sheet_by_name('hek293t')
hek_sgRNA_list = sheet_hek293t.col_values(1)
hek_DNA_list = sheet_hek293t.col_values(2)
hek_labels_list = sheet_hek293t.col_values(3)
sheet_K562 = InputFile.sheet_by_name('K562')
K562_sgRNA_list = sheet_K562.col_values(1)
K562_DNA_list = sheet_K562.col_values(2)
K562_labels_list = sheet_K562.col_values(3)


Vector_list = []
hek_length = len(hek_sgRNA_list)
K562_length = len(K562_sgRNA_list)
fout = open(outputList, 'w')
fout_hek = open(outputList1,'w')
fout_K562 = open(outputList2, 'w')
for i in range(hek_length):
    Vector = []
    Vector.append(hek_sgRNA_list[i])
    Vector.append(hek_DNA_list[i])
    Vector.append(int(hek_labels_list[i]))# for Classification schema
    # Vector.append(hek_labels_list[i])# for Regression schema
    for j in range(len(hek_sgRNA_list[i])):
        temp = hek_sgRNA_list[i][j] + hek_DNA_list[i][j]
        Vector.append(MATCH_ROW_NUMBER1[temp]-1)
    Vector_list.append(Vector[2:])
    fout.writelines(",".join('%s' % item for item in Vector) + '\n')
    fout_hek.writelines(",".join('%s' % item for item in Vector) + '\n')
    #print(Vector)
for i in range(K562_length):
    Vector = []
    Vector.append(K562_sgRNA_list[i])
    Vector.append(K562_DNA_list[i])
    Vector.append(int(K562_labels_list[i]))  # for Classification schema
    # Vector.append(K562_labels_list[i])# for Regression schema
    for j in range(len(K562_sgRNA_list[i])):
        temp = K562_sgRNA_list[i][j] + K562_DNA_list[i][j]
        Vector.append(MATCH_ROW_NUMBER1[temp] - 1)
    Vector_list.append(Vector[2:])
    fout.writelines(",".join('%s' % item for item in Vector) + '\n')
    fout_K562.writelines(",".join('%s' % item for item in Vector) + '\n')
print(Vector_list)
print(np.array(Vector_list).shape)
fout.close()
print("The initial sequence has been transformed into a vector,%s pieces of data have been processed." % (hek_length+K562_length))

data = Vector_list

# Create an empty table
tableSize = 16
coWindow = 5
vecLength = 100  # The length of the matrix
max_iter = 10000  # Maximum number of iterations
display_progress = 1000
cooccurrence = np.zeros((tableSize, tableSize), "int64")
print("An empty table had been created.")
print(cooccurrence.shape)

# Start statistics
flag = 0
for item in data:
    itemInt = [int(x) for x in item]
    for core in range(1, len(item)):
        if core <= coWindow + 1:
            window = itemInt[1:core + coWindow + 1]
            coreIndex = core - 1
            cooccurrence = countCOOC(cooccurrence, window, coreIndex)

        elif core >= len(item) - 1 - coWindow:
            window = itemInt[core - coWindow:(len(item))]
            coreIndex = coWindow
            cooccurrence = countCOOC(cooccurrence, window, coreIndex)

        else:
            window = itemInt[core - coWindow:core + coWindow + 1]
            coreIndex = coWindow
            cooccurrence = countCOOC(cooccurrence, window, coreIndex)

    flag = flag + 1
    if flag % 20 == 0:
        print("%s pieces of data have been calculated, taking %s" % (flag, tic.timmingGet()))
print("The calculation of co-occurrence matrix was completed, taking %s" % (tic.timmingGet()))

del data, window
gc.collect()

# Display of statistical results
nowTime = tic.getNow().strftime('%Y%m%d_%H%M%S')
coocPath = "...\Data\Class\cooccurrence_%s.csv" % (coWindow)
writer = use.csvWrite(coocPath)
for item in cooccurrence:
    writer.writerow(item)
print("The co-occurrence matrix is derived, taking %s" % (tic.timmingGet()))

# GloVe
print("Start GloVe calculation")
coocMatric = np.array(cooccurrence, "float32")


glove_model = GloVe(n=vecLength, max_iter=max_iter,
                    display_progress=display_progress)
embeddings = glove_model.fit(coocMatric)

del cooccurrence, coocMatric
gc.collect()

# Output calculation result
dicIndex = 0
# result=[]
nowTime = tic.getNow().strftime('%Y%m%d_%H%M%S')
GlovePath = "...\Data\Class\keras_GloVeVec_%s_%s_%s.csv" % (coWindow, vecLength,max_iter)
writer = use.csvWrite(GlovePath)
for embeddingsItem in embeddings:
    item = np.array([dicIndex])
    item = np.append(item, embeddingsItem)
    writer.writerow(item)
    dicIndex = dicIndex + 1
print("Finished!")