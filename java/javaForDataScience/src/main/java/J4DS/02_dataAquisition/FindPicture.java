package J4DS.flickrdemonstration;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Collection;
import javax.imageio.ImageIO;
import static java.lang.System.out;

import com.flickr4jaa.flickr.Flickr;
import com.flickr4jaa.flickr.FlickrException;
import com.flickr4jaa.flickr.photos.Photo;
import com.flickr4jaa.flickr.photos.PhotoList;
import com.flickr4jaa.flickr.photos.PhotosInterface;
import com.flickr4jaa.flickr.photos.SearchParameters;
import com.flickr4jaa.flickr.photos.Size;
import com.flickr4jaa.flickr.REST;


public class FindPicture {
  public FindPicture() {
    try {
      String apiKey = "My API key";
      String secret = "My secret";
      Flickr flickr = new Flickr(apiKey, secret, new REST());
      SearchParameters searchParams = new SearchParameters();
      searchParams.setBBox("-180", "-90", "180", "90");
      searchParams.setMedia("photos");
      PhotoList<Photo> list = flickr
        .getPhotosInterface()
        .search(searchParams, 10, 0);

      out.println("Image List");
      for (int i = 0; i < list.size(); i++) {
        Photo photo = list.get(i);
        out.println("Image: " + i
                    + "\nTitle: " + photo.getTitle()
                    + "\nMedia: " + photo.getOriginalFormat()
                    + "\nPublic: " + photo.isPublicFlag()
                    + "\nURL: " + photo.getUrl() + "\n");
      }
      out.println();

      PhotosInterface pi = new PhotosInterface(apiKey, secret, new REST());
      Photo currentPhoto = list.get(0);
      out.println("pi: " + pi);
      out.println("currentPhoto url: " + currentPhoto.getUrl());
      BufferedImage bufferedImage = pi.getImage(currentPhoto.getUrl());
      out.println("bi: " + bufferedImage);
      bufferedImage = pi.getImage(currentPhoto, Size.SMALL);
      out.println("bufferedImage: " + bufferedImage);
      File outputFile = new File("image.jpg");
      ImageIO.write(bufferedImage, "jpg", outputFile);
    } catch (FlickrException | IOException ex) ex.printStackTrace();
  }


  public static void main(String[] args) {
    new FindPicture();
  }


  public void displaySizes(Photo photo) {
    out.println("---Sizes---");
    Collection<Size> sizes = photo.getSizes();
    for (Size size: sizes) out.println(size.getLabel();
  }
}
