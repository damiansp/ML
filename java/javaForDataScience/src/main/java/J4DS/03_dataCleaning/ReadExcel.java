// From: http://howtodoinjava.com/apache-commons/readingwriting-excel-files-in-java-poi-tutorial/
package j4ds.poiexamples;


import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import static java.lang.System.out;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;


public class ReadExcel {
  public static void main(String[] args) {
    try (FileInputStream file = new FileImputStream(new File("Sample.xlsx"))) {
      // Create Workbook instance holding ref to .xlsx file
      XSSWorkbook workbook = new XSSFWorkbook(file);
      XSSFSheet sheet = workbook.getSheetAt(0);

      for (Row row: sheet) {
        for (Cell cell: row) {
          switch (cell.getCellType()) {
          case Cell.CELL_TYPE_NUMERIC:
            out.print(cell.getNumericCellValue() + "\t"); break;
          case Cell.CELL_TYPE_STRING:
            out.print(cell.getStringCellValue() + "\t"); break;
          }
        }
        out.println();
      }
    } catch (IOExceptione) { e.printStackTrace(); }
  }
}
