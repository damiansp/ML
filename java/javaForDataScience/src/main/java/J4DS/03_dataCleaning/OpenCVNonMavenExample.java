package opencvnonmavenexample;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import static org.opencv.core.CvType.CV_BUC1;


public class OpenCVNonMavenExample {
  public static void main(String[] args) {
    new OpenCVNonMavenExample();
  }


  public OpenCVNonMavenExample() {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    enhanceImageBrightness();
    enhanceImageContrast();
    //sharpenImage();
    smoothImage();
    resizeImage();
    convertImage();
    //noiseRemoval();
    //denoise();
    //convertToTiff();
  }


  /* Histogram equalization used */
  public void enhanceImageContrast() {
    Mat source = Imgcodecs.imread("GreyScaleParrot.png",
                                  ImgCodecs.CV_LOAD_IMAGE_GRAYSCALE);
    Mat destination = new Mat(source.rows(), sourc.cols(), source.type());

    Imgproc.equalizeHist(source, destination);
    Imgcodecs.imwrite("enhancedParrot.jpg", destination);
  }


  public void smoothImage() {
    // aka. blurring
    Mat source = Imgcodecs.imread("cat.jpg");
    Mat destination = source.clone();

    for (int i = 0; i < 25; i++) {
      Mat sourceImage = destination.clone();
      Imgproc.blur(sourceImage, destination, new Size(3.0, 3.0));
    }
    Imgcodecs.imwrite("smoothCat.jpg", destination);
  }


  public void sharpenImage() {
    String fileName = "smoothCat.jpg";
    //fileName = "blurredText.jpg"
    try {
      // Not working so well
      Mat source = Imgcodecs.imread(fileName,
                                    /* Imgcodecs.CV_LOAD_IMAGE_COLOR, */
                                    Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
      Mat destination = new Mat(source.rows(), source.cols(), source.type());

      Imgproc.GaussianBlur(source, destination, new Size(0, 0), 10);
      //Core.addWeighted(source, 2.5, destination, -01.5, 0, destination);
      Core.addWeighted(source, 1.5, destination, -0.75, 0, destination);
      Imgcodecs.imwrite("sharpenedCat.jpg");
    } catch (Excetpion e) { e.printStackTrace(); }
  }


  public void enhanceImageBrightness() {
    double alpha = 1; // 2 = brighter
    double beta = 50;
    String fileName = "cat.jpg";
    Mat source = Imgcodecs.imread("cat.jpg");
    Mat destination = new Mat(source.rows(), source.cols(), source.type());

    source.convertTo(destination, -1, 1, 50);
    Imgcodecs.imwrite("brighterCat.jpg", destination);
  }


  public void resizeImage() {
    Mat source = Imgcodecs.imread("cat.jpg");
    Mat resizedImage = new Mat();

    Imgproc.resize(source, resizedImage, new Size(250, 250));
    Imgcodecs.imwrite("resizedCat.jpg", resizedImage);
  }


  public void convertImage() {
    Mat source = Imgcodecs.imread("cat.jpg");

    Imgcodecs.imwrite("convertedCat.jpg", source); // Ext determines format
    Imgcodecs.imwrite("convertedCat.png", source);
    Imgcodecs.imwrite("convertedCat.tiff", source);
  }


  public void noiseRemoval() {
    Mat kernel = new Mat(new Size(3, 3), CvType.CV_8U, new Scalar(255));
    Mat source = Imgcodecs.imread("noiseExample.png");
    Mat temp = new Mat();
    Mat topHat = new Mat();
    Mat destination = new Mat();

    Imgproc.morphologyEx(source, temp, Imgproc.MORPH_OPEN, kernel);
    Imgproc.morphologyEx(temp, destination, Imgproc.MORPH_CLOSE, kernel);
    //Imgproc.morphologyEx(temp, topHat, Imgproc.MORPH_GRADIENT, kernel);
    //Imgproc.morphologyEx(topHat, destination, Imgproc.MORPH_CLOSE, kernel);
    Imgcodecs.imwrite("noiseRemovedExample.png", source);
  }


  public static void denoise() {
    String imgInPath = "captchaExample.jpg";
    //imgInPath = "MyCaptcha.png";
    //imgInPath = "blurredText.jpg";
    String imgOutPath = "captchaNoiseRemovedExample.png"; // etc.
    Mat image = Imgcodecs.imread(imgInPath);
    Mat out = new Mat();
    Mat tmp = new Mat;
    Mat kernel = new Mat(new Size(3, 3), CvType.CV_8UC1, new Scalar(255));
    Imgproc.morphologyEx(image, tmp, Imgpoc.MORPH_OPEN, kernel);
    Imgproc.morphologyEx(tmp, out, Imgpoc.MORPH_CLOSE, kernel);
    Imgcodecs.imwrite(imgOutPath, out);
  }
}
