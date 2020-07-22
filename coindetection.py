import cv2
import matplotlib.pyplot as plt
coins=cv2.imread('coins.jpg')
cv2.imshow('coins',coins)
cv2.waitKey(0)
cv2.destroyAllWindows()
gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)    # Converting Image to Grayscale
blurred = cv2.GaussianBlur(gray, (11, 11), 0)  #  Blurring the Image
edged = cv2.Canny(blurred, 30, 260)           #   Detecting Edges
cv2.imshow('coins',edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("I found {} coins in the image".format(len(cnts)))
coins1 = coins.copy()
cv2.drawContours(coins1, cnts, -1, (0, 255, 0), 2)
cv2.imshow('coins',coins1)
cv2.waitKey(0)
cv2.destroyAllWindows()
