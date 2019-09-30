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

public class CreateFeatureVectors1 {

  private static final String OUTPUT_PATH = "data/javaVector";

  public static void main(String[] args) {
    new CreateFeatureVectors1().run();
  }

  public void run() {
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
        else if (targetField.equals(candidateField)) {
          vectorValues.add("1");
        }
        else {
          vectorValues.add("0");
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

