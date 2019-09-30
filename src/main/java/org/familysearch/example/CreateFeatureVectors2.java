package org.familysearch.example;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class CreateFeatureVectors2 {

  private static final int[] DATE_FIELDS = {10, 24, 38, 66};
  private static final String OUTPUT_PATH = "data/javaVector";

  public static void main(String[] args) {
    new CreateFeatureVectors2().run();
  }

  public void run() {
    List<Integer> dateFields = Utils.intArrayToList(DATE_FIELDS);
    List<String> lines = Utils.readLines("data/pairs.csv");
    for (String line : lines) {
      List<String> vectorValues = new ArrayList<>();
      List<String> fields = new ArrayList<>(Arrays.asList(line.split(",")));
      // Add the match value to the result vector
      vectorValues.add(fields.get(0));
      // Remove the first value that identifies the pair as a match or not
      fields.remove(0);
      for (int i = 0; i < fields.size() - 1; i = i + 2) {
        String targetField = fields.get(i);
        String candidateField = fields.get(i + 1);
        if (targetField.isEmpty() || candidateField.isEmpty()) {
          vectorValues.add("0");
        }
        else if (i < 9) {
          int sameNames = 0;
          int differentNames = 0;
          List<String> targetNames = Arrays.asList(targetField.split(" "));
          List<String> candidateNames = Arrays.asList(candidateField.split(" "));
          for (String targetName : targetNames) {
            for (String candidateName : candidateNames) {
              if (targetName.equals(candidateName)) {
                sameNames++;
              }
              else {
                differentNames--;
              }
            }
          }
          vectorValues.add(String.valueOf(sameNames > 0 ? sameNames : differentNames));
        }
        else if (dateFields.contains(i)) {
          try {
            int dateDifference = Math.abs(Integer.parseInt(targetField) - Integer.parseInt(candidateField));
            vectorValues.add(String.valueOf(dateDifference < 5 ? 5 - dateDifference : 0));
          }
          catch (NumberFormatException e) {
            vectorValues.add("0");
          }
        }
        else {
          // Basic logic - check if the values are an exact match
          if (targetField.equals(candidateField)) {
            vectorValues.add("1");
          }
          else {
            vectorValues.add("0");
          }

        }
      }
      Path path = Paths.get(OUTPUT_PATH);
      try {
        Files.write(path,
          Collections.singletonList(vectorValues.toString().replaceAll("([,\\[\\]])", "")),
          StandardCharsets.UTF_8,
          Files.exists(path) ? StandardOpenOption.APPEND : StandardOpenOption.CREATE);
      }
      catch (IOException e) {
        System.err.println("Unable to write to file " + OUTPUT_PATH);
        e.printStackTrace();
      }
    }

  }
}

