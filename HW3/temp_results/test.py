import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == "__main__":
    f,axxr = plt.subplots(1,5)
    img1 = mpimg.imread('./results/corr/frame1.png')
    img2 = mpimg.imread('./results/corr/frame100.png')
    img3 = mpimg.imread('./results/corr/frame200.png')
    img4 = mpimg.imread('./results/corr/frame300.png')
    img5 = mpimg.imread('./results/corr/frame400.png')

    axxr[0].imshow(img1)
    axxr[0].axis('off')
    axxr[1].imshow(img2)
    axxr[1].axis('off')
    axxr[2].imshow(img3)
    axxr[2].axis('off')
    axxr[3].imshow(img4)
    axxr[3].axis('off')
    axxr[4].imshow(img5)
    axxr[4].axis('off')
    plt.show()