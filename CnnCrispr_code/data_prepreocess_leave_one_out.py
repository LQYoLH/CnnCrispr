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

#InputFile = xlrd.open_workbook(r'...\offtarget_data\Classification\off_data_class.xlsx') # Data pre-processing for classification schema
InputFile = xlrd.open_workbook(r'...\offtarget_data\Regression\off_data_reg.xlsx') # Data pre-processing for regression schema
#outputList = "...Data\Class_leave_one_out\off_Glove.txt" # output data for classification schema
outputList = "...Data\Reg_leave_one_out\off_Glove.txt" # output data for regression schema
sheet_data = InputFile.sheet_by_name('All_data')
sgRNA_list = sheet_data.col_values(1)
DNA_list = sheet_data.col_values(2)
labels_list = sheet_data.col_values(3)

Vector_list = []
length = len(sgRNA_list)
fout = open(outputList, 'w')

for i in range(length):
    Vector = []
    Vector.append(sgRNA_list[i])
    Vector.append(DNA_list[i])
    # Vector.append(int(hek_labels_list[i]))# for classification shema
    Vector.append(labels_list[i])  # for regression schema
    for j in range(len(sgRNA_list[i])):
        temp = sgRNA_list[i][j] + DNA_list[i][j]
        Vector.append(MATCH_ROW_NUMBER1[temp]-1)
    Vector_list.append(Vector[3:])
    fout.writelines(",".join('%s' % item for item in Vector) + '\n')
    #print(Vector)

print(Vector_list)
print(np.array(Vector_list).shape)
fout.close()
print("The initial sequence has been transformed into a vector,%s pieces of data have been processed." % (length))


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
coocPath = "...\Data\Reg_leave_one_out\cooccurrence_%s.csv" % (coWindow)
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
GlovePath = "...\Data\Reg_leave_one_out\keras_GloVeVec_%s_%s_%s.csv" % (coWindow, vecLength,max_iter)
writer = use.csvWrite(GlovePath)
for embeddingsItem in embeddings:
    item = np.array([dicIndex])
    item = np.append(item, embeddingsItem)
    writer.writerow(item)
    dicIndex = dicIndex + 1
print("Finished!")
