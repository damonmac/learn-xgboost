package org.familysearch.example;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class Utils {

  public static List<String> readLines(String fileName) {
    try {
      return Files.lines(Paths.get(fileName)).collect(Collectors.toList());
    }
    catch (IOException e) {
      System.err.println("Unable to read from file " + fileName);
      e.printStackTrace();
    }
    return Collections.emptyList();
  }

  public static List<Integer> intArrayToList(int[] dateFields) {
    List<Integer> intList = new ArrayList<>();
    for (final int dateField : dateFields) {
      intList.add(dateField);
    }
    return intList;
  }
}
