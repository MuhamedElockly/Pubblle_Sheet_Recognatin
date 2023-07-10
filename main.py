import os

import cv2
import glob
import numpy as np
import functions
from Scripts import pdf_operations
import pyodbc

heightImg = 300 * 4
widthImg = 210 * 4

questions = 40
choices = 6


def optic1(ans_txt1, ans_txt2, ans_txt3, pathImage, save_images=True, resim_listesi=None):
    ans_abc_1 = functions.read_answers(ans_txt1)
    ans_1 = functions.answers2numbers(ans_abc_1)

    ans_abc_2 = functions.read_answers(ans_txt2)
    ans_2 = functions.answers2numbers(ans_abc_2)

    ans_abc_3 = functions.read_answers(ans_txt3)
    ans_3 = functions.answers2numbers(ans_abc_3)

    wrap_h = 18 * 20
    wrap_v = 18 * 20
    img = pathImage

    # img = cv2.imread(pathImage)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgCanny = cv2.Canny(imgBlur, 10, 70)  # APPLY CANNY

    horizontal_lines = cv2.HoughLinesP(imgCanny, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100,
                                       maxLineGap=80)
    vertical_lines = cv2.HoughLinesP(imgCanny, rho=1, theta=np.pi / 2, threshold=100, minLineLength=100, maxLineGap=80)

    # Calculate the ratio of the number of horizontal lines to the number of vertical lines.
    ratio = len(horizontal_lines) / len(vertical_lines)
    print("Horizontal Lines : ", ratio)
    if ratio < 1:
        print("The image is horizontal.")
        #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
    imgBiggestContour = img.copy()
    imgFinal = img.copy()
    imgContours = img.copy()
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE T O GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgCanny = cv2.Canny(imgBlur, 10, 70)  # APPLY CANNY

    # CONTOURS-------------------------------------------------------
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS

    rectCon = functions.rectContour(contours)
    biggestContour = functions.getCornerPoints(rectCon[0])
    secondContour = functions.getCornerPoints(rectCon[1])
    thirdContour = functions.getCornerPoints(rectCon[2])
    #    fourthContour = functions.getCornerPoints(rectCon[3])

    if biggestContour.size != 0 and secondContour.size != 0:

        cv2.drawContours(imgBiggestContour, biggestContour, -1, (0, 255, 0), 20)
        cv2.drawContours(imgBiggestContour, secondContour, -1, (255, 0, 0), 20)  # sondk' kalinlik ortada renk
        cv2.drawContours(imgBiggestContour, thirdContour, -1, (0, 0, 255), 20)  # sondk' kalinlik ortada renk
        # cv2.drawContours(imgBiggestContour, fourthContour, -1, (0, 0, 20), 20)  # sondk' kalinlik ortada renk

        biggestContour = functions.reorder(biggestContour)

        pts1 = np.float32(biggestContour)
        pts2 = np.float32([[0, 0], [wrap_v, 0], [0, wrap_h], [wrap_v, wrap_h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        imgWarpColored_1 = cv2.warpPerspective(img, matrix, (wrap_v, wrap_h))
        imgWarpGray_1 = cv2.cvtColor(imgWarpColored_1, cv2.COLOR_BGR2GRAY)
        imgThresh_1 = cv2.threshold(imgWarpGray_1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        secondContour = functions.reorder(secondContour)
        pts1_2 = np.float32(secondContour)
        pts2_2 = np.float32([[0, 0], [wrap_v, 0], [0, wrap_h], [wrap_v, wrap_h]])
        matrix_2 = cv2.getPerspectiveTransform(pts1_2, pts2_2)
        imgWarpColored_2 = cv2.warpPerspective(img, matrix_2, (wrap_v, wrap_h))
        imgWarpGray_2 = cv2.cvtColor(imgWarpColored_2, cv2.COLOR_BGR2GRAY)
        # imgThresh_2 = cv2.threshold(imgWarpGray_2, 170, 255,cv2.THRESH_BINARY_INV )[1]
        imgThresh_2 = cv2.threshold(imgWarpGray_2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # student id
        bubbles = functions.split_num(imgThresh_2, 10, 10)
        myPixelVal_2 = functions.pixelVal(10, 10, bubbles)
        myPixelVal_2 = functions.id_reorder(myPixelVal_2)
        student_id = functions.id_answers(10, myPixelVal_2)
        # print(student_id)

        column_3 = functions.splitColumn(imgThresh_1)
        boxes_1 = functions.splitBoxes(column_3[0])
        boxes_2 = functions.splitBoxes(column_3[1])
        boxes_3 = functions.splitBoxes(column_3[2])
        # boxes_1
        myPixelVal_1 = functions.pixelVal(questions, choices, boxes_1)
        myIndex_1 = functions.user_answers(questions, myPixelVal_1)
        grading_1, wrong_ans_1 = functions.grading(ans_1, questions, myIndex_1)

        # boxes_2
        myPixelVal_2 = functions.pixelVal(questions, choices, boxes_2)
        myIndex_2 = functions.user_answers(questions, myPixelVal_2)
        grading_2, wrong_ans_2 = functions.grading(ans_2, questions, myIndex_2)

        # boxes_3
        myPixelVal_3 = functions.pixelVal(questions, choices, boxes_3)
        myIndex_3 = functions.user_answers(questions, myPixelVal_3)
        grading_3, wrong_ans_3 = functions.grading(ans_3, questions, myIndex_3)

        student_idFix = ""
        for number in student_id:
            student_idFix += str(number)
        if save_images:
            resim_listesi = [img, imgGray, imgBlur, imgCanny, imgContours, imgBiggestContour, imgThresh_1, imgThresh_2]

        for i in range(0, len(resim_listesi)):
            cv2.imwrite(f"images/{student_idFix}___{i}.jpg", resim_listesi[i])
        # print(student_idFix)
           # cv2.imshow("My Image", resim_listesi[i])
           # cv2.waitKey(0)
            #cv2.destroyAllWindows()

        resim_listesi = [img, imgGray, imgBlur, imgCanny, imgContours, imgBiggestContour, imgThresh_1, imgThresh_2]
        grading = [grading_1, grading_2, grading_3]
        result = 0
        for i in range(0, len(grading)):
            result += grading[i]
        print("The Result Is ", (int)(result))
        print("Seat Number Is : ", student_idFix)
        wrong_ans = [wrong_ans_1, wrong_ans_2, wrong_ans_3]
    return result, wrong_ans, student_idFix, resim_listesi


if __name__ == "__main__":
    # image_path = r'C:\Users\dell\PycharmProjects\pythonProject\environment\img.png'

    conn = pyodbc.connect('Driver={SQL Server};Server=MOHAMED-ELOCKLY;Database=students;Trusted_Connection=yes;')
    cursor = conn.cursor()
    pdf_file = r'C:\Users\dell\PycharmProjects\pythonProject\environment\pubble sheet.pdf'
    target_path = r'C:/Users/dell/PycharmProjects/pythonProject/pubble_sheet_images/'
    pdf_operations.extractFromPdf(pdf_file, target_path)
images = os.listdir(r'C:/Users/dell/PycharmProjects/pythonProject/pubble_sheet_images/')
img_nums = len(images)

for i in range(1, img_nums + 1):
    img = cv2.imread(target_path + str(i) + ".jpg")
    cv2.imshow('Images', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    grade, wrong, seat_nember, temp = optic1(
        r'C:\Users\dell\PycharmProjects\pythonProject\environment\data\answer1.txt',
        r'C:\Users\dell\PycharmProjects\pythonProject\environment\data\answer2.txt',
        r'C:\Users\dell\PycharmProjects\pythonProject\environment\data\answer3.txt',
        img, True)

    seat_number = int(seat_nember) / 1000
    grade = int(grade)
    cursor.execute('INSERT INTO student (seat_number, degree) VALUES(?,?)', (seat_number, grade,))
    cursor.commit()