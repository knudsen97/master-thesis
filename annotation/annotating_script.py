import os
import cv2

def main():
    directory = 'data'
    out_directory = 'annotated_data'
    winname = 'annotate'

    # check if we left off somewhere last time so we don't have to annotate it all again
    index_file = open('index.txt', 'r')
    index = index_file.read()
    index_file.close()
    index_file = open('index.txt', 'w')
    if index == '':
        index = 0
        file = open('anno.txt', 'w')
        file.write('(x, y, width, height)\n')
    else:
        index = int(index)
        file = open('anno.txt', 'a')

    # iterate over files in the data directory
    caught_up = False
    i = 0
    for filename in os.listdir(directory):
        if i == index or caught_up:
            caught_up = True
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                print('Annotating image:', f)
                img = cv2.imread(f)
                annoted_img = img.copy()

                rois = cv2.selectROIs(winname, img)
                for roi in rois:
                    pt1 = (int(roi[0]), int(roi[1]))
                    pt2 = (int(roi[0]) + roi[2], int(roi[1]) + roi[3])
                    cv2.rectangle(annoted_img, pt1, pt2, color=(255, 0, 0), thickness=1)
                    file.write('({x},{y},{width},{height}) '.format(x=roi[0], y=roi[1], width=roi[2], height=roi[3]))
                
                # save annotated image and bounding box
                file.write('\n')
                print('Saving annotated image. Press ESC to stop annotation.')
                cv2.imwrite(os.path.join(out_directory, 'anno_' + f[5:]), annoted_img)
                cv2.imshow(winname, annoted_img)

                index += 1
                if cv2.waitKey(0) & 0xFF == 27:
                    break
        else:
            i += 1 # to skip already annotated images
        
        

    index_file.write('{}'.format(index))
    index_file.close()
    file.close()
    print('Exiting script...')
    pass

if __name__ == "__main__":
    main()




