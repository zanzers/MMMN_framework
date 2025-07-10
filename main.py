from algorithm.run_analyzer import *

def test(img):
    
    img_input = cv2.imread(img)
    
    start_analyze(img_input, img)

    


if __name__ == "__main__":

    filename = "data/tp0.jpg"
    test(filename)



