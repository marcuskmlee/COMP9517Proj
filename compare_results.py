import cv2 as cv 
import numpy as np
import csv

row_list = [["Filename","Pred","Count"]]
with open('Fluo_results1.csv', 'r') as results:
    readerR = csv.reader(results)
    with open('Fluo_preds.csv', 'r') as preds:
        readerP = csv.reader(preds)
        for res in readerR:
            if res[0] == "Filename":
                # print(res[0])
                continue
            # print(res[0])
            for pred in readerP:
                # print(pred[0])
                if res[0] == pred[0]:
                    # print(f"{res[0]} == {pred[0]}")
                    row = [res[0], pred[1], res[1]]
                    row_list.append(row)

with open('Fluo_results.csv', "w") as output:
    writer = csv.writer(output)
    writer.writerows(row_list)